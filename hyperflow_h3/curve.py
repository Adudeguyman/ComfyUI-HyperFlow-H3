"""Experimental, checkpoint-bound eight-dimensional HyperFlow refits."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

import comfy.model_management
from comfy.patcher_extension import WrappersMP
from hyperflow_h3.embedder import (time_pairs, make_block_forward, make_final_forward,
                                   make_ctx_wrapper, _unwrap_two_time)
from hyperflow_h3.schedule import video_schedule_sigmas

_log = logging.getLogger("comfy.hyperflow")
FIT_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "curve_fits"


def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_fit(path):
    with safe_open(path, framework="pt", device="cpu") as handle:
        meta = handle.metadata() or {}
        if meta.get("format") != "hyperflow_curve_fit_v1":
            raise ValueError("unknown curve fit format")
        data = {key: handle.get_tensor(key) for key in ("generated", "pinned", "t", "r")}
    if (float(meta["strength"]) != 1.0 or not 0. <= float(meta["gate"]) <= 1.
            or float(meta["video_shift"]) != 12. or float(meta["audio_shift"]) != 3.):
        raise ValueError("unsupported curve fit recipe")
    for key in ("base_sha256", "adapter_sha256"):
        if len(meta[key]) != 64 or any(c not in "0123456789abcdef" for c in meta[key]):
            raise ValueError("invalid checkpoint hash in curve fit")
    for key, shape in (("generated", (16, 8)), ("pinned", (1025, 8)), ("t", (16,)), ("r", (16,))):
        if data[key].shape != shape or data[key].dtype != torch.float32 or not torch.isfinite(data[key]).all():
            raise ValueError(f"invalid {key} in curve fit")
    return data, meta


def matching_fit(model, weights, strength, gate_override, sigmas_override):
    """Tiered fit matching.

    Exact base+adapter hash -> silent. Same base but different adapter bytes,
    or a same-named (mirror/repack) base -> best-effort with a loud warning;
    the recipe (strength/gate/sigmas/shifts from the actual loaded files) must
    always match. Anything unknown keeps backbone-only, with hashes for support.
    A corrupt fit file never blocks the others."""
    reason = None
    if strength != 1.0 or gate_override is not None or sigmas_override:
        reason = "requires strength=1 and no gate/sigma overrides"
    elif model.patches or model.injections or any(
            key != "model_sampling" for key in model.object_patches):
        reason = ("requires an unmodified checkpoint (only a ModelSampling "
                  "node may precede ApplyHyperFlow)")
    elif not model.cached_patcher_init:
        reason = "checkpoint source is unavailable"
    else:
        source = model.cached_patcher_init[1][0]
        if not isinstance(source, (str, Path)) or not Path(source).is_file():
            reason = "checkpoint source is unavailable"
        else:
            try:
                base_hash, adapter_hash = file_hash(source), file_hash(weights.path)
                base_name = Path(source).name
                base_effort = name_effort = None
                for path in sorted(FIT_DIRECTORY.glob("*.safetensors")):
                    try:
                        data, meta = read_fit(path)
                    except (OSError, ValueError, TypeError, KeyError,
                            RuntimeError, SafetensorError):
                        continue  # unreadable fit must not block the others
                    if not (float(meta["strength"]) == strength
                            and float(meta["gate"]) == weights.metadata.gate
                            and torch.equal(video_schedule_sigmas(json.loads(meta["sigmas"])),
                                            video_schedule_sigmas(weights.metadata.sigmas))):
                        continue
                    base_exact = meta.get("base_sha256") == base_hash
                    adapter_exact = meta.get("adapter_sha256") == adapter_hash
                    if base_exact and adapter_exact:
                        return data, meta
                    if base_exact:
                        base_effort = base_effort or (data, meta, "adapter")
                    elif meta.get("base_name") == base_name:
                        name_effort = name_effort or (
                            data, meta, "base" if adapter_exact else "base and adapter")
                for candidate in (base_effort, name_effort):
                    if candidate is None:
                        continue
                    data, meta, what = candidate
                    _log.warning(
                        "[hyperflow] curve refit: the %s bytes differ from the "
                        "fitted checkpoint (mirror/repack?); the recipe matches, "
                        "so the fit is applied best-effort. Delete it from "
                        "assets/curve_fits if output looks wrong.", what)
                    return data, meta
                reason = (f"no fit matches this checkpoint, adapter and recipe "
                          f"(base sha256={base_hash}, adapter sha256={adapter_hash})")
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, SafetensorError) as exc:
                reason = f"fit unavailable ({exc})"
    _log.warning("[hyperflow] curve refit disabled: %s; using backbone only.", reason)
    return None


def table_lookup(table, t):
    pos = t.clamp(0., 1.) * (len(table) - 1)
    i = pos.floor().long().clamp(max=len(table) - 2)
    return torch.lerp(table[i], table[i + 1], (pos - i).unsqueeze(1))


def install_curve_refit(model, fit, verbose=False):
    data, meta = fit
    dm = model.get_model_object("diffusion_model")
    patch_key = "diffusion_model.time_embedder"
    if patch_key in model.object_patches:
        raise RuntimeError("This MODEL already has HyperFlow applied; chain it once.")
    shared = {}
    grid = video_schedule_sigmas(json.loads(meta["sigmas"]))
    warned = False

    def warn():
        nonlocal warned
        if not warned:
            warned = True
            _log.warning("[hyperflow] curve refit disabled for an unmatched sampling recipe; using backbone only.")

    def embed(t_vals):
        ctx = shared["ctx"]
        table = comfy.model_management.cast_to(dm.adaln_t_table, device=t_vals.device)
        if not ctx["curve_active"]:
            out = table_lookup(table, t_vals)
        else:
            t, r = time_pairs(t_vals, shared)
            pinned = data["pinned"].to(t_vals.device)
            out = table_lookup(pinned, t)
            # A pair is either one of the 16 trained intervals or pinned (t,t).
            for i, (pt, pr) in enumerate(zip(data["t"].tolist(), data["r"].tolist())):
                take = ((t - pt).abs() < 1e-6) & ((r - pr).abs() < 1e-6) & (t != r)
                out[take] = data["generated"][i].to(out.device)
        # Native non-curve dispatch casts to compute dtype; consumers below
        # retain the original fp32 curve coordinates, as native curves do.
        ctx["curve_embedding"] = out
        return out

    def wrap_consumer(base, remapped):
        def forward(x, t_emb, *args, **kwargs):
            ctx = shared["ctx"]
            return (remapped if ctx["curve_active"] else base)(
                x, ctx["curve_embedding"], *args, **kwargs)
        forward._hyperflow_two_time = True
        forward._hyperflow_base_forward = base
        return forward

    context_wrapper = make_ctx_wrapper(shared, dm, grid, "curve-refit", verbose)

    def wrap(executor, *args, **kwargs):
        options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        options = options or {}
        sigmas = options.get("sample_sigmas")
        valid = (sigmas is not None and sigmas.shape == grid.shape
                 and torch.equal(sigmas.cpu().float(), grid)
                 and float(options.get("minimax_h3_sigma_shift_video", dm.sigma_shift_video)) == float(meta["video_shift"])
                 and float(options.get("minimax_h3_sigma_shift_audio", dm.sigma_shift_audio)) == float(meta["audio_shift"]))
        if not valid:
            warn()
            shared["ctx"] = {"curve_active": False}
            try:
                return executor(*args, **kwargs)
            finally:
                shared["ctx"] = None

        def run(*inner_args, **inner_kwargs):
            ctx = shared["ctx"]
            # Exact float32 pairs mirror the native schedule arithmetic. Custom
            # samplers evaluating intermediate times must not consume this fit.
            pair_v = (ctx["t_v"], ctx["r_v"])
            pair_a = (ctx["t_a"], ctx["r_a"])
            pairs = list(zip(data["t"].tolist(), data["r"].tolist()))
            def matches(pair, choices):
                return any(abs(pair[0] - t) < 1e-6 and abs(pair[1] - r) < 1e-6 for t, r in choices)
            ctx["curve_active"] = matches(pair_v, pairs[:8]) and matches(pair_a, pairs[8:])
            if not ctx["curve_active"]:
                warn()
            return executor(*inner_args, **inner_kwargs)

        return context_wrapper(run, *args, **kwargs)

    model.add_object_patch("diffusion_model.use_adaln_curves", False)
    # A callable attribute, not a new nn.Module: streaming state_dict is unchanged.
    model.add_object_patch(patch_key, embed)
    for i, block in enumerate(dm.blocks):
        base = _unwrap_two_time(block.forward)
        model.add_object_patch(f"diffusion_model.blocks.{i}.forward",
                               wrap_consumer(base, make_block_forward(base, shared)))
    base = _unwrap_two_time(dm.final_layer.forward)
    model.add_object_patch("diffusion_model.final_layer.forward",
                           wrap_consumer(base, make_final_forward(base, shared)))
    model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "hyperflow_curve_refit", wrap)
    return float(meta["gate"])
