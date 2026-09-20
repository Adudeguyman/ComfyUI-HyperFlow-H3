import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import comfy.model_management
import comfy.utils
from comfy.ldm.minimax.model import MiniMaxH3Model, time_shift_sigma
from comfy.model_sampling import ModelSamplingAV
from comfy.weight_adapter import LoRAAdapter
from hyperflow_h3.apply import apply_lora, _bypass_adapter, _SplitQKVLoRA, _FrugalLoRA
from hyperflow_h3.embedder import install_two_time
from hyperflow_h3.schedule import video_schedule_sigmas
from hyperflow_h3.weights import _download, load_weights
from test_apply import _ApplyPatcher, _LoRAWeights
from test_embedder import (_FakePatcher, _FakeWeights, _embedder, _endpoint_tensors,
                           _manual_endpoint, ALPHA, GATE, FREQ, HID, OUT)
from test_weights import HEADER


@pytest.mark.parametrize("layout", [None, "gate_value"])
def test_fc1_conversion_matches_diffusers(layout, tmp_path):
    torch.manual_seed(5)
    a, b = torch.randn(4, 6), torch.randn(16, 4)
    header = dict(HEADER)
    stored = b
    if layout:
        header["hyperflow_fc1_layout"] = layout
        stored = torch.cat(b.chunk(2)[::-1])
    path = tmp_path / "node.safetensors"
    save_file({"blocks.0.mlp.fc1.lora_A.weight": a,
               "blocks.0.mlp.fc1.lora_B.weight": stored}, str(path), metadata=header)
    weights = load_weights(path)
    x = torch.randn(3, 6)
    value, gate = F.linear(F.linear(x, a), b).chunk(2, dim=-1)
    converted = F.linear(F.linear(x, weights.lora["blocks.0.mlp.fc1"][0]),
                         weights.lora["blocks.0.mlp.fc1"][1])
    new_gate, new_value = converted.chunk(2, dim=-1)
    torch.testing.assert_close(new_value * F.silu(new_gate), value * F.silu(gate))


@pytest.mark.parametrize("variant", ["full", "pruned"])
def test_converter_marks_corrected_rows(tmp_path, variant):
    spec = importlib.util.spec_from_file_location("convert_hyperflow", Path(__file__).parents[1] / "tools/convert_hyperflow.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    source, dest = tmp_path / "original.safetensors", tmp_path / "converted.safetensors"
    a, b = torch.randn(4, 6), torch.randn(16, 4)
    save_file({"transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight": a,
               "transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight": b}, str(source), metadata=HEADER)
    converter.convert(source, dest, variant)
    weights = load_weights(dest)
    assert weights.metadata.raw["hyperflow_fc1_layout"] == "gate_value"
    assert torch.equal(weights.lora["blocks.0.mlp.fc1"][1], torch.cat(b.chunk(2)[::-1]))


def test_download_creates_directory_and_preserves_cache(tmp_path, monkeypatch):
    cached = tmp_path / "cache.safetensors"
    cached.write_bytes(b"complete model")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(cached))
    result = _download("repo", "weights.safetensors", tmp_path / "models/hyperflow")
    assert result.read_bytes() == cached.read_bytes()
    assert not list(result.parent.glob("*.part"))


def test_download_follows_relative_cache_symlink(tmp_path, monkeypatch):
    blob = tmp_path / "blob"
    blob.write_bytes(b"model")
    cache = tmp_path / "snapshot"
    cache.mkdir()
    link = cache / "model.safetensors"
    try:
        link.symlink_to(Path("../blob"))
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(link))
    result = _download("repo", "weights.safetensors", tmp_path / "models")
    assert result.read_bytes() == b"model"
    assert not result.is_symlink()
    assert link.is_file()


def test_failed_copy_does_not_publish_partial_download(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: "cache")
    def fail(source, dest):
        Path(dest).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr("hyperflow_h3.weights.shutil.copyfile", fail)
    with pytest.raises(OSError, match="disk full"):
        _download("repo", "weights.safetensors", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compact_qkv_matches_separate_upstream_branches(dtype):
    torch.manual_seed(3)
    a = torch.randn(12, 16, dtype=dtype)
    parts = [torch.randn(8, 4, dtype=dtype) for _ in range(3)]
    b = torch.block_diag(*parts)
    adapter = _bypass_adapter(LoRAAdapter(set(), (b, a, 12., None, None, None)), "blocks.0.attn.qkv_proj")
    assert isinstance(adapter, _SplitQKVLoRA)
    assert adapter.weights[0].numel() == b.numel() // 3
    x = torch.randn(9, 16, dtype=dtype)
    base = torch.randn(9, 24, dtype=dtype)
    expected = base + torch.cat([F.linear(F.linear(x, down), up)
                                for down, up in zip(a.chunk(3), parts)], dim=-1)
    assert torch.equal(adapter.bypass_forward(lambda x: base.clone(), x), expected)
    b[0, -1] = 1
    assert isinstance(_bypass_adapter(LoRAAdapter(set(), (b, a, 12., None, None, None)),
                                     "blocks.0.attn.qkv_proj"), _FrugalLoRA)


def test_merge_keeps_base_time_lora_out_of_endpoint_weights(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    te = _embedder()
    dm = torch.nn.Module()
    dm.time_embedder = te
    patcher = _ApplyPatcher(dm)
    endpoint = _endpoint_tensors()
    original = {name: p.detach().clone() for name, p in te.named_parameters()}
    weights = _LoRAWeights({"time_embedder." + k: v for k, v in endpoint.items()},
                          {"time_embedder." + k: ALPHA for k in endpoint})
    apply_lora(patcher, weights, 1., "merge")
    assert not patcher.patches
    injection = patcher.injections["hyperflow_lora"][0]
    injection.inject(patcher)
    try:
        r = torch.tensor([0.2, 0.7])
        torch.testing.assert_close(te(r), _manual_endpoint(te, endpoint, r))
        for name, p in te.named_parameters():
            assert torch.equal(p, original[name])
    finally:
        injection.eject(patcher)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("workflow", ["t2va", "fl2va", "ref2va"])
def test_real_model_routes_pairs_through_blocks_and_output_heads(masked, workflow):
    torch.manual_seed(6)
    dm = MiniMaxH3Model(hidden_size=12, num_attention_heads=2, attention_head_dim=6,
                       num_layers=1, token_refiner_num_layers=0, ffn_hidden_size=16,
                       latents_dim=1, audio_latents_dim=2, text_dim=12,
                       timestep_input_dim=FREQ, time_embed_hidden_size=HID, time_embed_dim=OUT,
                       rope_inv_freq_len=1, dtype=torch.float32, operations=torch.nn)
    dm.rope.inv_freq.fill_(1.)
    dm.requires_grad_(False)
    endpoint = _endpoint_tensors()
    original_te = dm.time_embedder.forward
    captured = {}
    original_block, original_final = dm.blocks[0].forward, dm.final_layer.forward
    def block(x, temb, segments, rope, *args, **kwargs):
        captured["block"] = (temb, segments)
        return original_block(x, temb, segments, rope, *args, **kwargs)
    def final(x, temb, video, audio, *args, **kwargs):
        captured["final"] = (temb, video, audio)
        return original_final(x, temb, video, audio, *args, **kwargs)
    dm.blocks[0].forward, dm.final_layer.forward = block, final
    patcher = _FakePatcher(dm)
    install_two_time(patcher, _FakeWeights(endpoint, GATE))
    for key, value in patcher.object_patches.items():
        comfy.utils.set_attr(dm, key.removeprefix("diffusion_model."), value)
    video, audio = torch.randn(1, 1, 1, 2, 4), torch.randn(1, 2, 2, 1)
    context = torch.randn(1, 2, 12)
    grid = video_schedule_sigmas(None)
    payload = {"text_token_tags": torch.tensor([0, 2])}
    if workflow == "fl2va":
        payload.update(keyframes=[{"resolved_frame_index": 0, "latent": video.clone()}],
                       cond_video_latents=[video.clone()])
    elif workflow == "ref2va":
        payload.update(refs=[{"kind": "image", "latent_h": 2, "latent_w": 4},
                             {"kind": "audio", "ref_audio_t": 1}],
                       cond_video_latents=[video.clone()], cond_audio_latents=[audio.clone()])
    masks = {"denoise_mask": torch.tensor([[[[[0., 0., 1., 1.], [0., 0., 1., 1.]]]]]),
             "audio_denoise_mask": torch.tensor([[[[0.], [1.]]]])} if masked else {}
    for step, sigma in enumerate(grid[:-1]):
        options = {"sample_sigmas": grid}
        output = patcher.wrappers[0](dm._forward, [video, audio], sigma.reshape(1) * 1000.,
                                     context, options, minimax_payload=payload, **masks)
        assert all(torch.isfinite(x).all() for x in output)
        # Independent row plan: target audio has its own clock even at t=0.
        tv = float(1. - (sigma * 1000. / 1000.).clamp(min=1e-6))
        ta = float(1. - time_shift_sigma((sigma * 1000. / 1000.).clamp(min=1e-6), 12., 3.))
        rv, ra = float(1. - grid[step + 1]), float(1. - time_shift_sigma(grid[step + 1], 12., 3.))
        def expected(t, r):
            base = original_te(torch.tensor([t]))
            return (base + GATE * (_manual_endpoint(dm.time_embedder, endpoint, torch.tensor([r])) - base))[0]
        temb, video_seg, audio_seg = captured["final"]
        if masked:
            torch.testing.assert_close(temb[audio_seg[2]], torch.stack([expected(1., 1.), expected(ta, ra)]))
            torch.testing.assert_close(temb[video_seg[2]], torch.stack([expected(.999, .999), expected(tv, rv)]))
        else:
            torch.testing.assert_close(temb[audio_seg[2]], expected(ta, ra))
            torch.testing.assert_close(temb[video_seg[2]], expected(tv, rv))
        layout = options["minimax_h3_layout"]
        for start, stop, row in captured["block"][1]:
            kind = next(kind for a, b, kind in layout.segments if a <= start < b)
            if kind == "audio":
                torch.testing.assert_close(temb[row // 3], temb[audio_seg[2]])
            elif kind in ("cond", "ref_img"):
                torch.testing.assert_close(temb[row // 3], expected(.999, .999))
            elif kind == "ref_audio":
                torch.testing.assert_close(temb[row // 3], expected(1., 1.))
            elif kind == "text":
                torch.testing.assert_close(temb[row // 3], expected(tv, rv))


def test_sparse_recipe_starts_on_third_evaluation():
    config = SimpleNamespace(sampling_settings={"shift": 12., "audio_shift": 3.})
    threshold = ModelSamplingAV(config).percent_to_sigma(.16)
    assert (video_schedule_sigmas(None)[:-1] > threshold).tolist() == [True, True] + [False] * 6
