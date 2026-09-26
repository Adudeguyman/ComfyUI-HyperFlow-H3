"""Fit HyperFlow two-time conditioning to a pruned/curve MiniMax-H3 base.

Pruned bases replace the time embedder with an 8-wide curve table, so the
HyperFlow (t, r) pathway has nowhere to go. This fits, per AdaLN projection
(blocks + final layer), the 8-vector whose output best matches what the
full-weight teacher produces with HyperFlow's time-embedder LoRAs, for the 16
trained (t, r) pairs and a 1025-point pinned (t, t) curve. The result binds to
the exact pruned base and node adapter files (SHA-256); curve.py applies it.
Used by the Create HyperFlow H3 Curve Fit node and tools/fit_curve.py; see CURVE_FITS.md.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy.ldm.minimax.model import time_shift_sigma

from .curve import file_hash
from .schedule import video_schedule_sigmas

_log = logging.getLogger("comfy.hyperflow")
PINNED_GRID = 1025


def linear_weights(handle, prefix, device):
    weight = handle.get_tensor(prefix + ".weight").to(device)
    if weight.dtype == torch.int8:
        config = json.loads(bytes(handle.get_tensor(prefix + ".comfy_quant").tolist()))
        if config["format"] != "int8_tensorwise":
            raise ValueError(f"{prefix}: unsupported quantization {config['format']!r}; "
                             "use a bf16/fp16 or int8_tensorwise teacher")
        params = TensorWiseINT8Layout.Params(
            scale=handle.get_tensor(prefix + ".weight_scale").to(device),
            orig_dtype=torch.float32, orig_shape=tuple(weight.shape),
            convrot=config.get("convrot", False),
            convrot_groupsize=config.get("convrot_groupsize", 256))
        weight = TensorWiseINT8Layout.dequantize(weight, params)
    return weight.float(), handle.get_tensor(prefix + ".bias").to(device).float()


def time_loras(handle, device):
    """time_embedder / endpoint_time_embedder LoRAs from either the original
    diffusers-layout file or the converted full node build."""
    loras = {}
    for key in handle.keys():
        name = key.removeprefix("transformer.").replace(".linear_1.", ".proj_in.").replace(".linear_2.", ".proj_out.")
        if name.startswith(("time_embedder.", "endpoint_time_embedder.")):
            loras[name] = handle.get_tensor(key).to(device).float()
    if not loras:
        raise ValueError("the HyperFlow file has no time_embedder LoRAs; use the original "
                         "file or the converted *full* build, not the pruned one")
    return loras


def fit_filename(pruned_path):
    return Path(pruned_path).stem + "_hyperflow_curvefit.safetensors"


def open_checkpoint(path):
    """(handle, key prefix): full checkpoints nest the model under model.diffusion_model."""
    handle = safe_open(path, framework="pt")
    prefix = "model.diffusion_model." if any(k.startswith("model.diffusion_model.") for k in handle.keys()) else ""
    return handle, prefix


def parse_splice(spec):
    """'FILE:FIRST-LAST' (or 'FILE:BLOCK') -> (FILE, FIRST, LAST)."""
    name, _, blocks = spec.strip().rpartition(":")
    first, _, last = blocks.partition("-")
    if not name or not first.isdigit() or not (last or first).isdigit():
        raise ValueError(f"splice {spec!r}: expected FILE:FIRST-LAST, e.g. minimax_h3_ref2va_int8_convrot.safetensors:20-49")
    return name, int(first), int(last or first)


def fit_curve(pruned_path, teacher_path, splices, hyperflow_path, adapter_path, output, device, progress=None):
    """splices: [(path, first_block, last_block)] teacher overrides for hybrids.
    Writes the fit to ``output`` and returns a text report."""
    (pruned, pp), teacher = open_checkpoint(pruned_path), open_checkpoint(teacher_path)
    if pp + "adaln_t_table" not in pruned.keys():
        raise ValueError(f"{Path(pruned_path).name} is not a pruned/curve MiniMax-H3 base (no adaln_t_table)")
    if teacher[1] + "time_embedder.proj_in.weight" not in teacher[0].keys():
        raise ValueError(f"{Path(teacher_path).name} is not a full-weight MiniMax-H3 base (no time_embedder)")
    donors = [(range(first, last + 1), open_checkpoint(path)) for path, first, last in splices]

    with safe_open(hyperflow_path, framework="pt") as hyperflow:
        metadata = hyperflow.metadata() or {}
        if metadata.get("hyperflow") != "true":
            raise ValueError(f"{Path(hyperflow_path).name} is not a HyperFlow weights file")
        loras = time_loras(hyperflow, device)
    alpha, gate = float(metadata["lora_alpha"]), float(metadata["hyperflow_gate"])
    video_shift = float(metadata.get("hyperflow_video_shift", 12.))
    audio_shift = float(metadata.get("hyperflow_audio_shift", 3.))
    time_weights = {p: linear_weights(teacher[0], teacher[1] + "time_embedder." + p, device) for p in ("proj_in", "proj_out")}

    def embed(t, branch=None):
        # comfy.ldm.minimax.model.TimeEmbedder, plus an optional LoRA branch
        half = time_weights["proj_in"][0].shape[1] // 2
        freqs = torch.exp(-math.log(10000.) * torch.arange(half, device=device) / half)
        x = torch.cat(((t[:, None] * freqs).cos(), (t[:, None] * freqs).sin()), -1)
        for p in ("proj_in", "proj_out"):
            y = F.linear(x, *time_weights[p])
            if branch:
                a, b = loras[f"{branch}.{p}.lora_A.weight"], loras[f"{branch}.{p}.lora_B.weight"]
                y = y + F.linear(F.linear(x, a), b) * (alpha / a.shape[0])
            x = F.silu(y) if p == "proj_in" else y
        return x

    # The same (t, r) arithmetic the runtime compares against (see curve.py).
    grid = video_schedule_sigmas(json.loads(metadata["hyperflow_sigmas"]))
    current = (grid[:-1] * 1000. / 1000.).clamp(min=1e-6)
    t = torch.cat((1. - current, 1. - time_shift_sigma(current, video_shift, audio_shift)))
    r = torch.cat((1. - grid[1:], 1. - time_shift_sigma(grid[1:], video_shift, audio_shift)))
    tt, rr, pt = (x.to(device) for x in (t, r, torch.linspace(0., 1., PINNED_GRID)))
    targets = torch.cat((F.silu((1. - gate) * embed(tt, "time_embedder") + gate * embed(rr, "endpoint_time_embedder")),
                         F.silu((1. - gate) * embed(pt, "time_embedder") + gate * embed(pt, "endpoint_time_embedder"))))

    table = pruned.get_tensor(pp + "adaln_t_table").to(device)
    pos = tt.clamp(0., 1.) * (len(table) - 1)
    i = pos.floor().long().clamp(max=len(table) - 2)
    backbone = torch.lerp(table[i], table[i + 1], (pos - i).unsqueeze(1))

    blocks = 1 + max(int(k.removeprefix(pp).split(".")[1]) for k in pruned.keys() if k.startswith(pp + "blocks."))
    names = [f"blocks.{b}.adaln_proj.linear" for b in range(blocks)] + ["final_layer.adaln_proj.linear"]
    fits, lines, errors = [], [], []
    for b, name in enumerate(names):
        source, sp = next((h for span, h in donors if b in span), teacher)
        fw, fb = linear_weights(source, sp + name, device)
        cw, cb = linear_weights(pruned, pp + name, device)
        target = F.linear(targets, fw, fb)
        del fw
        # least squares over the 8 curve coordinates, per target row
        x = torch.linalg.lstsq(cw.double(), (target - cb).double().T).solution.T.float()
        fits.append(x.cpu())
        generated = target[:16]
        before, after = (((F.linear(emb, cw, cb) - generated).norm(dim=1) / generated.norm(dim=1)).mean().item()
                         for emb in (backbone, x[:16]))
        errors.append((before, after))
        lines.append(f"{name}: backbone-only {before:.2%}  refit {after:.2%}")
        _log.info("[hyperflow] curve fit %s", lines[-1])
        del cw, target, x
        if progress is not None:
            progress()

    fits = torch.stack(fits)
    backbone_mean = sum(e[0] for e in errors) / len(errors)
    refit_mean = sum(e[1] for e in errors) / len(errors)
    teachers = [Path(teacher_path).name] + [f"{Path(path).name}:{first}-{last}" for path, first, last in splices]
    binding = {
        "format": "hyperflow_curve_fit_v1", "base_name": Path(pruned_path).name,
        "base_sha256": file_hash(pruned_path), "adapter_sha256": file_hash(adapter_path),
        "teacher": json.dumps(teachers), "gate": str(gate), "strength": "1.0",
        "sigmas": metadata["hyperflow_sigmas"], "video_shift": str(video_shift), "audio_shift": str(audio_shift),
        "backbone_error": f"{backbone_mean:.6f}", "refit_error": f"{refit_mean:.6f}",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"generated": fits[:, :16].contiguous(), "pinned": fits[:, 16:].contiguous(),
               "t": t.float().contiguous(), "r": r.float().contiguous()}, output, metadata=binding)
    summary = (f"mean relative modulation error: backbone-only {backbone_mean:.2%} -> refit {refit_mean:.2%}\n"
               f"wrote {output}")
    _log.info("[hyperflow] curve fit %s", summary.replace("\n", "; "))
    return "\n".join(lines + [summary])
