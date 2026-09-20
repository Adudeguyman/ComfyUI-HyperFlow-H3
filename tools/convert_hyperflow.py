"""Convert a HyperFlow weights file to ComfyUI module paths on disk.

Reads the original diffusers-layout file (repo id layout: transformer.<module>.
lora_{A,B}.weight) and writes a NEW .safetensors -- the original stays untouched
-- whose keys are ComfyUI MiniMax-H3 module paths (<module>.lora_A/lora_B.weight)
with the to_q/to_k/to_v triples fused into per attn.qkv_proj rank-3r pairs. The
HyperFlow header is preserved, with a layout marker for the swapped SwiGLU
rows. Gate, sigmas and rank remain exactly as in the original.

Two builds, matching the two base-model families:

* full     -- every key, incl. time_embedder + endpoint_time_embedder (the
              two-time conditioning); for full (non-pruned) MiniMax-H3 bases.
* pruned   -- backbone only (DiT blocks + token refiner); for pruned/curve
              MiniMax-H3 bases, which have no time_embedder. Runs single-time:
              off-recipe, output deviates from the released model.

Usage (from the node pack root, with the ComfyUI venv active):

    python tools/convert_hyperflow.py minimax_h3_hyperflow_8step_v1.0.safetensors
    python tools/convert_hyperflow.py path/to/dir --variant pruned -o out/

A hyperflow.json manifest is written next to the outputs and printed, for the
copy bundled with the node pack (users then only download the .safetensors).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from hyperflow_h3.keys import fused_qkv_pair, plan_from_keys
from hyperflow_h3.weights import read_metadata, resolve_weights

FULL_SUFFIX = "_comfyui"
PRUNED_SUFFIX = "_comfyui_pruned"


def convert(src: Path, dst: Path, variant: str) -> int:
    metadata = read_metadata(src)
    with safe_open(str(src), framework="pt", device="cpu") as handle:
        plan = plan_from_keys(handle.keys())
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    out = {}
    n_qkv = 0
    for target, entry in plan.items():
        is_time = target.startswith("time_embedder.") \
            or target.startswith("endpoint_time_embedder.")
        if variant == "pruned" and is_time:
            continue
        if entry["kind"] == "qkv":
            a3, b3 = fused_qkv_pair(entry, tensors)
            out[f"{target}.lora_A.weight"] = a3
            out[f"{target}.lora_B.weight"] = b3
            n_qkv += 1
        else:
            out[f"{target}.lora_A.weight"] = tensors[entry["A"]]
            b = tensors[entry["B"]]
            if target.endswith(".mlp.fc1") and metadata.raw.get("hyperflow_fc1_layout") != "gate_value":
                value, gate = b.chunk(2, dim=0)
                b = torch.cat((gate, value), dim=0)
            out[f"{target}.lora_B.weight"] = b

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst), metadata={**metadata.raw, "hyperflow_fc1_layout": "gate_value"})
    n = len([k for k in out if k.endswith(".lora_A.weight")])
    print(f"[convert] {variant}: {n} LoRA modules ({n_qkv} fused qkv) -> {dst}")
    return n


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="original HyperFlow .safetensors or directory")
    parser.add_argument("-o", "--output", help="output file (single variant) or "
                        "directory (default: next to the input)")
    parser.add_argument("--variant", choices=["full", "pruned", "both"], default="both")
    args = parser.parse_args(argv)

    src = resolve_weights(args.input)
    stem = src.stem
    variants = ["full", "pruned"] if args.variant == "both" else [args.variant]

    outputs = []
    for variant in variants:
        suffix = FULL_SUFFIX if variant == "full" else PRUNED_SUFFIX
        if args.output:
            out = Path(args.output)
            if out.suffix != ".safetensors" or len(variants) > 1:
                out = out / f"{stem}{suffix}.safetensors"
        else:
            out = src.with_name(f"{stem}{suffix}.safetensors")
        convert(src, out, variant)
        outputs.append(out)

    for out, variant in zip(outputs, variants):
        manifest = {"default": out.name, "weights": {out.name: {
            "hyperflow_version": read_metadata(out).version,
            "sha256": sha256_of(out), "base": variant}}}
        manifest_name = "hyperflow.json" if variant == "full" else "hyperflow_pruned.json"
        manifest_path = outputs[0].with_name(manifest_name)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[convert] manifest ({variant}) -> {manifest_path}")
        print(f"--- paste into the node pack's assets/{manifest_name} ---")
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
