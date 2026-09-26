"""Compare curve-coordinate conditioning with the full HyperFlow pathway.

Reads one AdaLN projection at a time; never loads the backbone. Run with the
ComfyUI venv, --full/--pruned/--adapter paths and --output report.json.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout


def linear_weights(handle, prefix, device):
    weight = handle.get_tensor(prefix + ".weight").to(device)
    if weight.dtype == torch.int8:
        config = json.loads(bytes(handle.get_tensor(prefix + ".comfy_quant").tolist()))
        assert config["format"] == "int8_tensorwise", config
        params = TensorWiseINT8Layout.Params(
            scale=handle.get_tensor(prefix + ".weight_scale").to(device),
            orig_dtype=torch.float32, orig_shape=tuple(weight.shape),
            convrot=config.get("convrot", False),
            convrot_groupsize=config.get("convrot_groupsize", 256))
        weight = TensorWiseINT8Layout.dequantize(weight, params)
    return weight.float(), handle.get_tensor(prefix + ".bias").to(device).float()


def errors(actual, target):
    return {
        "max_abs": (actual - target).abs().amax(-1).tolist(),
        "relative_l2": ((actual - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-12)).tolist(),
        "cosine": F.cosine_similarity(actual, target, dim=-1).tolist(),
    }


def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(args):
    torch.set_num_threads(8)
    device = args.device
    with safe_open(args.full, framework="pt") as full, safe_open(args.pruned, framework="pt") as curve, safe_open(args.adapter, framework="pt") as adapter:
        time_weights = {p: linear_weights(full, "time_embedder." + p, device)
                        for p in ("proj_in", "proj_out")}
        metadata = adapter.metadata()
        alpha = float(metadata["lora_alpha"])
        gate = float(metadata["hyperflow_gate"])
        loras = {k: adapter.get_tensor(k).to(device).float() for k in adapter.keys()
                 if k.startswith(("time_embedder.", "endpoint_time_embedder."))}

        def embed(t, branch=None):
            half = time_weights["proj_in"][0].shape[1] // 2
            freqs = torch.exp(-math.log(10000.) * torch.arange(half, device=device) / half)
            phase = t[:, None] * freqs
            x = torch.cat((phase.cos(), phase.sin()), -1)
            for p in ("proj_in", "proj_out"):
                y = F.linear(x, *time_weights[p])
                if branch:
                    a = loras[f"{branch}.{p}.lora_A.weight"]
                    b = loras[f"{branch}.{p}.lora_B.weight"]
                    y = y + F.linear(F.linear(x, a), b) * (alpha / a.shape[0])
                x = F.silu(y) if p == "proj_in" else y
            return x

        table = curve.get_tensor("adaln_t_table").to(device)

        def lookup(t):
            pos = t.clamp(0., 1.) * (len(table) - 1)
            i = pos.floor().long().clamp(max=len(table) - 2)
            return torch.lerp(table[i], table[i + 1], (pos - i).unsqueeze(1))

        raw = torch.tensor(json.loads(metadata["hyperflow_sigmas"]), device=device)
        sv = 12. * raw / (1. + 11. * raw)
        # Same video->audio conversion and timestep round trip as native _forward.
        current = (sv[:-1] * 1000. / 1000.).clamp(min=1e-6)
        tv, rv = 1. - current, 1. - sv[1:]
        def audio_time(sigma):
            base = sigma / (12. + sigma * (1. - 12.))
            return 1. - 3. * base / (1. + (3. - 1.) * base)
        ta, ra = audio_time(current), audio_time(sv[1:])
        t, r = torch.cat((tv, ta)), torch.cat((rv, ra))
        coordinate = (1. - gate) * t + gate * r
        target = F.silu((1. - gate) * embed(t, "time_embedder") + gate * embed(r, "endpoint_time_embedder"))
        pinned_t = torch.linspace(0., 1., 1025, device=device)
        pinned_target = F.silu((1. - gate) * embed(pinned_t, "time_embedder")
                               + gate * embed(pinned_t, "endpoint_time_embedder"))
        targets = torch.cat((target, pinned_target))
        raw_t = F.silu(embed(t))
        ideal_coordinate = F.silu(embed(coordinate))
        curve_t, curve_coordinate = lookup(t), lookup(coordinate)
        curve_blend = torch.lerp(lookup(t), lookup(r), gate)
        report = {
            "files": {k: str(getattr(args, k)) for k in ("full", "pruned", "adapter")},
            "math": "fp32 projections of dequantized checkpoint weights, no activation quantization; rows video steps 1-8 then audio steps 1-8",
            "table_shape": list(table.shape), "gate": gate,
            "t": t.tolist(), "r": r.tolist(), "coordinate": coordinate.tolist(),
            "endpoint_vs_raw": errors(embed(r, "endpoint_time_embedder"), embed(r)),
            "endpoint_vs_adapted": errors(embed(r, "endpoint_time_embedder"), embed(r, "time_embedder")),
            "coordinate_embedding_vs_blend": errors(embed(coordinate), (1. - gate) * embed(t, "time_embedder") + gate * embed(r, "endpoint_time_embedder")),
            "projections": {},
        }
        # Best shared eight-coordinate fit is an optimistic bound for Option 2:
        # it may use arbitrary coordinates, not just points on the time curve.
        gram = torch.zeros(8, 8, dtype=torch.float64, device=device)
        rhs = torch.zeros(8, len(targets), dtype=torch.float64, device=device)
        fit_stats = {}
        for name in [f"blocks.{i}.adaln_proj.linear" for i in range(50)] + ["final_layer.adaln_proj.linear"]:
            fw, fb = linear_weights(full, name, device)
            cw, cb = linear_weights(curve, name, device)
            all_intended = F.linear(targets, fw, fb)
            intended = all_intended[:16]
            baseline = F.linear(curve_t, cw, cb)
            actual = F.linear(curve_coordinate, cw, cb)
            blend = F.linear(curve_blend, cw, cb)
            raw_mod = F.linear(raw_t, fw, fb)
            ideal_mod = F.linear(ideal_coordinate, fw, fb)
            modalities, expand = (1, 2) if name.startswith("final") else (3, 6)
            def split(x):
                return x.view(16, modalities, expand, -1)
            w = cw.double().view(modalities, expand, -1, 8)
            y = (all_intended - cb).view(len(targets), modalities, expand, -1).double()
            group_gram = torch.einsum("mehi,mehj->meij", w, w)
            group_rhs = torch.einsum("mehi,smeh->smei", w, y)
            gram += group_gram.sum((0, 1))
            rhs += group_rhs.sum((1, 2)).T
            all_norm = all_intended.view(len(targets), modalities, expand, -1).double().square().sum(-1)
            fit_stats[name] = (group_gram, group_rhs, y.square().sum(-1), all_norm,
                               split(intended - raw_mod).double().square().sum(-1))
            report["projections"][name] = {
                "coordinate_vs_target": errors(split(actual), split(intended)),
                "backbone_only_vs_target": errors(split(baseline), split(intended)),
                "table_blend_vs_target": errors(split(blend), split(intended)),
                "pruning_vs_raw": errors(split(baseline), split(raw_mod)),
                "ideal_coordinate_vs_target": errors(split(ideal_mod), split(intended)),
                "coordinate_delta_vs_target_delta": errors(split(actual - raw_mod), split(intended - raw_mod)),
                "backbone_delta_vs_target_delta": errors(split(baseline - raw_mod), split(intended - raw_mod)),
            }
            del fw, fb, cw, cb, y, all_intended
            print(name, "coordinate relL2", ((actual-intended).norm()/intended.norm()).item(), flush=True)
        fitted = torch.linalg.solve(gram, rhs).T
        report["refit_coordinates"] = fitted[:16].tolist()
        report["pinned_grid_size"] = len(pinned_t)
        for name, (gg, gr, yy, norm, delta_norm) in fit_stats.items():
            sse = (torch.einsum("si,meij,sj->sme", fitted, gg, fitted)
                   - 2 * torch.einsum("si,smei->sme", fitted, gr) + yy).clamp_min(0)
            report["projections"][name]["refit_vs_target"] = {
                "relative_l2": (sse[:16] / norm[:16].clamp_min(1e-24)).sqrt().tolist(),
                "relative_delta_l2": (sse[:16] / delta_norm.clamp_min(1e-24)).sqrt().tolist()}
            pinned_error = (sse[16:] / norm[16:].clamp_min(1e-24)).sqrt()
            report["projections"][name]["pinned_refit_vs_target"] = {
                "mean_relative_l2_per_group": pinned_error.mean(0).tolist(),
                "max_relative_l2_per_group": pinned_error.amax(0).tolist()}
        def mean_error(names, key):
            values = [torch.tensor(report["projections"][n][key]["relative_l2"]).flatten() for n in names]
            return torch.cat(values).mean().item()
        blocks = [n for n in report["projections"] if n.startswith("blocks")]
        final = ["final_layer.adaln_proj.linear"]
        report["go_no_go"] = {label: {key: mean_error(names, key) for key in
            ("backbone_only_vs_target", "coordinate_vs_target", "refit_vs_target")}
            for label, names in (("blocks", blocks), ("final_layer", final), ("all_groups", blocks + final))}
        if args.fit:
            binding = {
                "format": "hyperflow_curve_fit_v1", "base_sha256": file_hash(args.pruned),
                "teacher_sha256": file_hash(args.full), "adapter_sha256": file_hash(args.pruned_adapter),
                "teacher_adapter_sha256": file_hash(args.adapter), "base_name": Path(args.pruned).name,
                "gate": str(gate), "strength": "1.0", "sigmas": metadata["hyperflow_sigmas"],
                "video_shift": "12.0", "audio_shift": "3.0",
            }
            save_file({"generated": fitted[:16].float().cpu().contiguous(),
                       "pinned": fitted[16:].float().cpu().contiguous(),
                       "t": t.cpu(), "r": r.cpu()}, args.fit, metadata=binding)
            report["fit_metadata"] = binding
        Path(args.output).write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full", "pruned", "adapter", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit", help="Export the checkpoint-bound experimental fit")
    parser.add_argument("--pruned-adapter", help="Exact backbone adapter file to bind the export to")
    run(parser.parse_args())
