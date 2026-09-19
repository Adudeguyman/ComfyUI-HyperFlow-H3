"""Translate HyperFlow (diffusers/PEFT) LoRA keys onto ComfyUI MiniMax-H3 module paths.

Two on-disk layouts are accepted, detected per file:

* original -- `transformer.<module>.lora_{A,B}.weight` in diffusers naming;
* converted -- `<comfy_module>.lora_{A,B}.weight` in ComfyUI naming, as written by
  tools/convert_hyperflow.py (q/k/v already fused into the attn.qkv_proj pairs).

ComfyUI's native MiniMax-H3 tree differs from diffusers in three ways this module
absorbs:

* names: `transformer_blocks.{i}` -> `blocks.{i}`, `ff.net.0.proj` -> `mlp.fc1`,
  `time_embedder.linear_1` -> `time_embedder.proj_in`, refiner blocks nest under
  `token_refiner.blocks` (not `refiner_blocks`);
* structure: diffusers keeps `attn.to_q/to_k/to_v` separate, ComfyUI fuses them into
  one `attn.qkv_proj` -- the three LoRA pairs fuse into a single rank-3r update
  (A' = cat(A_q, A_k, A_v), B' = block_diag(B_q, B_k, B_v); the 3x alpha correction
  is applied by weights.py, which owns per-entry alpha);
* `endpoint_time_embedder.*` has no native module: HyperFlow's endpoint copy of the
  time embedder is held as captured tensors by the two-time forward patch
  (hyperflow_h3.embedder), never registered on the module tree.

Anything that does not line up is an error, never a warning -- same policy as the
reference loader (src/hyperflow_h3/lora.py on the diffusers side).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch

KEY_RE = re.compile(r"^transformer\.(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$")

#: converted-layout keys are ComfyUI module paths under these families
DIRECT_RE = re.compile(
    r"^(?P<module>(?:blocks\.\d+|token_refiner\.blocks\.\d+)\."
    r"(?:attn\.(?:qkv_proj|out_proj)|mlp\.fc[12])"
    r"|time_embedder\.proj_(?:in|out)"
    r"|endpoint_time_embedder\.proj_(?:in|out))"
    r"\.lora_(?P<matrix>[AB])\.weight$")

# diffusers module path -> (kind, comfy path or qkv group path, part)
_RULES = [
    (re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_(q|k|v)$"), "qkv",
     lambda m: f"blocks.{m[1]}.attn.qkv_proj", lambda m: m[2]),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn\.to_out\.0$"), "lora",
     lambda m: f"blocks.{m[1]}.attn.out_proj", None),
    (re.compile(r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj$"), "lora",
     lambda m: f"blocks.{m[1]}.mlp.fc1", None),
    (re.compile(r"^transformer_blocks\.(\d+)\.ff\.net\.2$"), "lora",
     lambda m: f"blocks.{m[1]}.mlp.fc2", None),
    (re.compile(r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_(q|k|v)$"), "qkv",
     lambda m: f"token_refiner.blocks.{m[1]}.attn.qkv_proj", lambda m: m[2]),
    (re.compile(r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0$"), "lora",
     lambda m: f"token_refiner.blocks.{m[1]}.attn.out_proj", None),
    (re.compile(r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0\.proj$"), "lora",
     lambda m: f"token_refiner.blocks.{m[1]}.mlp.fc1", None),
    (re.compile(r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2$"), "lora",
     lambda m: f"token_refiner.blocks.{m[1]}.mlp.fc2", None),
    (re.compile(r"^time_embedder\.linear_1$"), "lora",
     lambda m: "time_embedder.proj_in", None),
    (re.compile(r"^time_embedder\.linear_2$"), "lora",
     lambda m: "time_embedder.proj_out", None),
    (re.compile(r"^endpoint_time_embedder\.linear_1$"), "endpoint",
     lambda m: "endpoint_time_embedder.proj_in", None),
    (re.compile(r"^endpoint_time_embedder\.linear_2$"), "endpoint",
     lambda m: "endpoint_time_embedder.proj_out", None),
]

# converted layout: identity mapping, same plan shape as the translated rules
_DIRECT_ENDPOINT = {"endpoint_time_embedder.proj_in", "endpoint_time_embedder.proj_out"}


def _translate(module: str):
    for regex, kind, path_fn, part_fn in _RULES:
        m = regex.match(module)
        if m is not None:
            return kind, path_fn(m), part_fn(m) if part_fn else None
    return None


def plan_from_keys(keys: Iterable[str]) -> dict:
    """Group the file's keys by ComfyUI target module, auto-detecting the layout.

    Returns a plan dict:
      {comfy_module: {"kind": "lora", "A": key, "B": key}}
      {comfy_module: {"kind": "qkv", "parts": {"q": {"A": key, "B": key}, ...}}}
      {"endpoint_time_embedder.proj_in":  {"kind": "endpoint", "A": key, "B": key}, ...}
    Raises on keys of neither layout, on unknown modules, and on modules carrying
    only one of lora_A / lora_B.
    """
    plan: dict = {}
    for key in keys:
        match = KEY_RE.match(key)
        if match is not None:
            kind_path = _translate(match["module"])
            if kind_path is None:
                raise ValueError(
                    f"{key!r} targets a module this MiniMax-H3 port does not know; the file "
                    "does not match the expected HyperFlow layout.")
            kind, path, part = kind_path
        else:
            match = DIRECT_RE.match(key)
            if match is None:
                raise ValueError(
                    f"Unexpected key {key!r}. Expected HyperFlow keys "
                    "(`transformer.<module>.lora_A.weight`, diffusers layout) or converted "
                    "ComfyUI keys (`<module>.lora_A.weight`).")
            path = match["module"]
            kind = "endpoint" if path in _DIRECT_ENDPOINT else "lora"
            part = None

        matrix = match["matrix"]
        if kind == "qkv":
            entry = plan.setdefault(path, {"kind": kind, "parts": {}})
            slot = entry["parts"].setdefault(part, {})
        else:
            entry = plan.setdefault(path, {"kind": kind})
            slot = entry
        if entry["kind"] != kind:
            raise ValueError(f"{key!r} conflicts with another key over {path}.")
        if matrix in slot:
            raise ValueError(f"Duplicate {key!r} in the weights file.")
        slot[matrix] = key

    incomplete = []
    for path, entry in plan.items():
        if entry["kind"] == "qkv":
            for part, slot in entry["parts"].items():
                if set(slot) != {"A", "B"}:
                    incomplete.append(f"{path}.{part}")
        elif set(entry) - {"kind"} != {"A", "B"}:
            incomplete.append(path)
    if incomplete:
        raise ValueError(f"Modules with only one of lora_A / lora_B: {incomplete[:8]}")
    return plan


def fused_qkv_pair(plan_entry: dict, tensors: dict):
    """Build the single rank-3r pair (A', B') for a fused qkv_proj from the three
    to_q/to_k/to_v LoRA pairs. Output order matches Attention.forward's split:
    [q | k | v] contiguous thirds. The caller must pair this with alpha' = 3x the
    file alpha so the effective scale stays alpha/rank per part."""
    parts = plan_entry["parts"]
    order = ("q", "k", "v")
    if set(parts) != set(order):
        raise ValueError(f"A fused qkv module needs to_q/to_k/to_v pairs, got {sorted(parts)}.")
    a_q, a_k, a_v = (tensors[parts[p]["A"]] for p in order)
    b_q, b_k, b_v = (tensors[parts[p]["B"]] for p in order)
    rank = a_q.shape[0]
    if any(t.shape[0] != rank for t in (a_k, a_v)):
        raise ValueError("to_q/to_k/to_v LoRA ranks disagree; cannot fuse.")
    a3 = torch.cat([a_q, a_k, a_v], dim=0)
    inner = b_q.shape[0]
    if any(t.shape != (inner, rank) for t in (b_k, b_v)):
        raise ValueError("to_q/to_k/to_v LoRA output shapes disagree; cannot fuse.")
    b3 = torch.zeros((3 * inner, 3 * rank), dtype=b_q.dtype)
    for i, b in enumerate((b_q, b_k, b_v)):
        b3[i * inner:(i + 1) * inner, i * rank:(i + 1) * rank] = b
    return a3.contiguous(), b3.contiguous()


__all__ = ["DIRECT_RE", "KEY_RE", "fused_qkv_pair", "plan_from_keys"]
