import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import comfy.utils
from comfy.ldm.minimax.model import MiniMaxH3Model, time_shift_sigma
from comfy.model_patcher import ModelPatcher
from hyperflow_h3.curve import (file_hash, install_curve_refit, matching_fit,
                                table_lookup)
from hyperflow_h3.nodes import ApplyHyperFlow, ApplyHyperFlowAdvanced
from hyperflow_h3.schedule import DEFAULT_SIGMAS_8STEP, video_schedule_sigmas
from test_embedder import _FakePatcher


def fixture():
    torch.manual_seed(19)
    dm = MiniMaxH3Model(hidden_size=12, num_attention_heads=2, attention_head_dim=6,
                       num_layers=1, token_refiner_num_layers=0, ffn_hidden_size=16,
                       latents_dim=1, audio_latents_dim=2, text_dim=12,
                       time_embed_dim=8, adaln_curve_grid=1025,
                       rope_inv_freq_len=1, dtype=torch.float32, operations=torch.nn)
    dm.rope.inv_freq.fill_(1.)
    dm.adaln_t_table.copy_(torch.randn(1025, 8) * .1)
    dm.requires_grad_(False)
    grid = video_schedule_sigmas(None)
    current = (grid[:-1] * 1000. / 1000.).clamp(min=1e-6)
    t = torch.cat((1. - current, 1. - time_shift_sigma(current, 12., 3.)))
    r = torch.cat((1. - grid[1:], 1. - time_shift_sigma(grid[1:], 12., 3.)))
    data = {"t": t, "r": r, "generated": torch.randn(16, 8) * .1,
            "pinned": torch.randn(1025, 8) * .1}
    meta = {"format": "hyperflow_curve_fit_v1", "gate": "0.25", "strength": "1.0",
            "sigmas": json.dumps(DEFAULT_SIGMAS_8STEP), "video_shift": "12.0", "audio_shift": "3.0"}
    return dm, (data, meta), grid


@pytest.mark.parametrize("workflow", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("masked", [False, True])
def test_curve_routes_generated_pinned_and_masked_rows(workflow, masked):
    dm, fit, grid = fixture()
    data, _ = fit
    captured = {}
    block, final = dm.blocks[0].forward, dm.final_layer.forward

    def capture_block(x, emb, segments, *args, **kwargs):
        captured["block"] = (emb, segments)
        return block(x, emb, segments, *args, **kwargs)

    def capture_final(x, emb, video, audio, *args, **kwargs):
        captured["final"] = (emb, video, audio)
        return final(x, emb, video, audio, *args, **kwargs)

    dm.blocks[0].forward, dm.final_layer.forward = capture_block, capture_final
    patcher = _FakePatcher(dm)
    install_curve_refit(patcher, fit)
    state_keys = set(dm.state_dict())
    for key, value in patcher.object_patches.items():
        comfy.utils.set_attr(dm, key.removeprefix("diffusion_model."), value)
    assert set(dm.state_dict()) == state_keys
    video, audio = torch.randn(1, 1, 1, 2, 4), torch.randn(1, 2, 2, 1)
    context = torch.randn(1, 2, 12)
    payload = {"text_token_tags": torch.tensor([0, 2]),
               "visual_cond_noise_aug": 0., "audio_cond_noise_aug": 0.}
    if workflow == "fl2va":
        payload.update(keyframes=[{"resolved_frame_index": 0, "latent": video}],
                       cond_video_latents=[video])
    if workflow == "ref2va":
        payload.update(refs=[{"kind": "image", "latent_h": 2, "latent_w": 4},
                             {"kind": "audio", "ref_audio_t": 1}],
                       cond_video_latents=[video], cond_audio_latents=[audio])
    masks = {"denoise_mask": torch.tensor([[[[[.5, .5, 1., 1.], [.5, .5, 1., 1.]]]]]),
             "audio_denoise_mask": torch.tensor([[[[0.], [1.]]]])} if masked else {}
    for step, sigma in enumerate(grid[:-1]):
        options = {"sample_sigmas": grid}
        out = patcher.wrappers[0](dm._forward, [video, audio], sigma.reshape(1) * 1000.,
                                  context, options, minimax_payload=payload, **masks)
        assert all(torch.isfinite(x).all() for x in out)
        emb, v, a = captured["final"]
        assert emb.dtype == torch.float32
        expected_v, expected_a = data["generated"][step], data["generated"][8 + step]
        if masked:
            pin_v = table_lookup(data["pinned"], torch.tensor([1. - .5 * (sigma * 1000. / 1000.)]))[0]
            torch.testing.assert_close(emb[v[2]], torch.stack((pin_v, expected_v)))
            torch.testing.assert_close(emb[a[2]], torch.stack((data["pinned"][-1], expected_a)))
        else:
            torch.testing.assert_close(emb[v[2]], expected_v)
            torch.testing.assert_close(emb[a[2]], expected_a)
        for start, _, row in captured["block"][1]:
            kind = next(k for x, y, k in options["minimax_h3_layout"].segments if x <= start < y)
            if kind == "text":
                torch.testing.assert_close(emb[row // 3], expected_v)
            elif kind in ("cond", "ref_img", "ref_audio"):
                t = data["t"][step + (8 if kind == "ref_audio" else 0)]
                torch.testing.assert_close(emb[row // 3], table_lookup(data["pinned"], t.reshape(1))[0])
        if step == 0:
            assert not torch.equal(expected_v, expected_a)


@pytest.mark.parametrize("mismatch", ["missing", "schedule", "shift", "time"])
def test_runtime_mismatch_is_bit_exact_backbone_fallback(mismatch, caplog):
    dm, fit, grid = fixture()
    original = copy.deepcopy(dm)
    patcher = _FakePatcher(dm)
    install_curve_refit(patcher, fit)
    for key, value in patcher.object_patches.items():
        comfy.utils.set_attr(dm, key.removeprefix("diffusion_model."), value)
    x = [torch.randn(1, 1, 1, 2, 4), torch.randn(1, 2, 2, 1)]
    text = torch.randn(1, 2, 12)
    options = {"sample_sigmas": grid}
    timestep = grid[0].reshape(1) * 1000.
    if mismatch == "missing": options = {}
    if mismatch == "schedule": options["sample_sigmas"] = torch.tensor([1., .5, 0.])
    if mismatch == "shift": options["minimax_h3_sigma_shift_audio"] = 4.
    if mismatch == "time": timestep = torch.tensor([500.])
    want = original._forward(x, timestep, text, dict(options))
    for _ in range(2):
        got = patcher.wrappers[0](dm._forward, x, timestep, text, dict(options))
        assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert sum("curve refit disabled" in r.message for r in caplog.records) == 1


def test_model_patcher_restores_curve_attributes_and_reinstall():
    dm, fit, _ = fixture()
    holder = torch.nn.Module()
    holder.diffusion_model = dm
    model = ModelPatcher(holder, torch.device("cpu"), torch.device("cpu"))
    for _ in range(2):
        clone = model.clone()
        install_curve_refit(clone, fit)
        clone.patch_model(load_weights=False)
        assert not dm.use_adaln_curves
        assert callable(dm.time_embedder)
        clone.unpatch_model(unpatch_weights=False)
        assert dm.use_adaln_curves
        assert not hasattr(dm, "time_embedder")


@pytest.mark.parametrize("mismatch", [None, "base", "strength", "gate", "sigmas", "corrupt",
                                       "fit_gate", "fit_shift", "fit_sigmas", "fit_shape", "fit_nan"])
def test_fit_binding_and_recipe_fallback(tmp_path, monkeypatch, caplog, mismatch):
    # NB "adapter" is intentionally absent: a byte-different adapter on an
    # exact base now applies best-effort (covered by the test below).
    import hyperflow_h3.curve as curve
    _, (data, meta), _ = fixture()
    checkpoint, adapter = tmp_path / "base.bin", tmp_path / "adapter.bin"
    checkpoint.write_bytes(b"specific checkpoint")
    adapter.write_bytes(b"specific adapter")
    meta.update(base_sha256=file_hash(checkpoint), adapter_sha256=file_hash(adapter))
    if mismatch == "fit_gate": meta["gate"] = "0.5"
    if mismatch == "fit_shift": meta["audio_shift"] = "4.0"
    if mismatch == "fit_sigmas": meta["sigmas"] = "[1, 0.5, 0]"
    if mismatch == "fit_shape": data["generated"] = data["generated"][:8].contiguous()
    if mismatch == "fit_nan": data["pinned"][0, 0] = float("nan")
    path = tmp_path / "fit.safetensors"
    save_file(data, path, metadata=meta)
    monkeypatch.setattr(curve, "FIT_DIRECTORY", tmp_path)
    model = SimpleNamespace(patches={}, object_patches={}, injections={}, cached_patcher_init=(None, (checkpoint,)))
    weights = SimpleNamespace(path=adapter, metadata=SimpleNamespace(gate=.25, sigmas=DEFAULT_SIGMAS_8STEP))
    if mismatch == "base": checkpoint.write_bytes(b"different checkpoint")
    if mismatch == "corrupt": path.write_bytes(b"broken")
    fit = matching_fit(model, weights, .8 if mismatch == "strength" else 1.,
                       .25 if mismatch == "gate" else None, "1,0" if mismatch == "sigmas" else None)
    assert (fit is None) == (mismatch is not None)
    assert sum("using backbone only" in r.message for r in caplog.records) == int(mismatch is not None)


def test_fit_name_match_applies_best_effort_with_warning(tmp_path, monkeypatch, caplog):
    """Tiered matching: exact base + different adapter bytes (mirror/HF copy)
    applies best-effort; a same-named repacked base likewise; unknown names
    keep the backbone-only fallback with hashes in the log. A ModelSampling
    object patch is the only allowed pre-existing patch."""
    import hyperflow_h3.curve as curve
    _, (data, meta), _ = fixture()
    checkpoint = tmp_path / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    adapter = tmp_path / "adapter.bin"
    checkpoint.write_bytes(b"original checkpoint")
    adapter.write_bytes(b"specific adapter")
    meta.update(base_name=checkpoint.name,
                base_sha256=file_hash(checkpoint), adapter_sha256=file_hash(adapter))
    save_file(data, tmp_path / "fit.safetensors", metadata=meta)
    monkeypatch.setattr(curve, "FIT_DIRECTORY", tmp_path)
    model = SimpleNamespace(patches={}, object_patches={}, injections={},
                            cached_patcher_init=(None, (checkpoint,)))
    weights = SimpleNamespace(path=adapter, metadata=SimpleNamespace(gate=.25, sigmas=DEFAULT_SIGMAS_8STEP))

    # Exact base, byte-different adapter (the common HF-mirror case): applies.
    adapter.write_bytes(b"adapter copy from another source")
    fit = matching_fit(model, weights, 1., None, None)
    assert fit is not None
    assert sum("best-effort" in r.message for r in caplog.records) == 1

    # A ModelSampling node before ApplyHyperFlow is allowed...
    model.object_patches = {"model_sampling": object()}
    assert matching_fit(model, weights, 1., None, None) is not None
    # ...anything else patching the model first is not.
    model.object_patches = {"diffusion_model.blocks.0.forward": object()}
    assert matching_fit(model, weights, 1., None, None) is None
    model.object_patches = {}

    # Same-named repacked base: applies best-effort too.
    adapter.write_bytes(b"specific adapter")
    checkpoint.write_bytes(b"repacked mirror with different bytes")
    assert file_hash(checkpoint) != meta["base_sha256"]
    fit = matching_fit(model, weights, 1., None, None)
    assert fit is not None
    assert sum("best-effort" in r.message for r in caplog.records) == 3

    # An unknown checkpoint name still falls back, with hashes for support.
    other = tmp_path / "some_other_pruned.safetensors"
    other.write_bytes(b"unknown checkpoint")
    model.cached_patcher_init = (None, (other,))
    fit = matching_fit(model, weights, 1., None, None)
    assert fit is None
    assert any("using backbone only" in r.message and file_hash(other) in r.message
               for r in caplog.records)


def test_curve_toggle_defaults_off():
    for node in (ApplyHyperFlow, ApplyHyperFlowAdvanced):
        assert node.INPUT_TYPES()["optional"]["experimental_curve_refit"][1]["default"] is False


def test_curve_rejects_double_install():
    dm, fit, _ = fixture()
    patcher = _FakePatcher(dm)
    install_curve_refit(patcher, fit)
    with pytest.raises(RuntimeError, match="already has HyperFlow"):
        install_curve_refit(patcher, fit)
