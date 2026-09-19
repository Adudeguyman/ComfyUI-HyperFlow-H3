"""Key-plan tests: converted layout, original layout, fusion math, errors."""
import sys
from pathlib import Path

import pytest
import torch

_COMFYUI_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_COMFYUI_ROOT))
sys.path.insert(0, str(_PACKAGE))

from hyperflow_h3.keys import fused_qkv_pair, plan_from_keys

CONVERTED = [
    "blocks.0.attn.qkv_proj.lora_A.weight",
    "blocks.0.attn.qkv_proj.lora_B.weight",
    "blocks.0.attn.out_proj.lora_A.weight",
    "blocks.0.attn.out_proj.lora_B.weight",
    "blocks.0.mlp.fc1.lora_A.weight",
    "blocks.0.mlp.fc1.lora_B.weight",
    "blocks.0.mlp.fc2.lora_A.weight",
    "blocks.0.mlp.fc2.lora_B.weight",
    "token_refiner.blocks.0.attn.qkv_proj.lora_A.weight",
    "token_refiner.blocks.0.attn.qkv_proj.lora_B.weight",
    "token_refiner.blocks.0.mlp.fc2.lora_A.weight",
    "token_refiner.blocks.0.mlp.fc2.lora_B.weight",
    "time_embedder.proj_in.lora_A.weight",
    "time_embedder.proj_in.lora_B.weight",
    "time_embedder.proj_out.lora_A.weight",
    "time_embedder.proj_out.lora_B.weight",
    "endpoint_time_embedder.proj_in.lora_A.weight",
    "endpoint_time_embedder.proj_in.lora_B.weight",
    "endpoint_time_embedder.proj_out.lora_A.weight",
    "endpoint_time_embedder.proj_out.lora_B.weight",
]


def test_converted_plan_kinds():
    plan = plan_from_keys(CONVERTED)
    assert plan["blocks.0.attn.qkv_proj"]["kind"] == "lora"
    assert plan["blocks.0.attn.out_proj"]["kind"] == "lora"
    assert plan["time_embedder.proj_in"]["kind"] == "lora"
    assert plan["endpoint_time_embedder.proj_in"]["kind"] == "endpoint"
    assert set(plan) == {k.rsplit(".lora_", 1)[0] for k in CONVERTED}
    for entry in plan.values():
        slot = entry if entry["kind"] == "lora" else entry
        assert set(slot) - {"kind"} == {"A", "B"}


def test_original_plan_groups_qkv():
    keys = [f"transformer.transformer_blocks.2.attn.to_{p}.lora_{m}.weight"
            for p in ("q", "k", "v") for m in ("A", "B")]
    plan = plan_from_keys(keys)
    entry = plan["blocks.2.attn.qkv_proj"]
    assert entry["kind"] == "qkv"
    assert set(entry["parts"]) == {"q", "k", "v"}
    assert set(entry["parts"]["q"]) == {"A", "B"}


def test_original_plan_translates_names():
    keys = [
        "transformer.transformer_blocks.3.ff.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.3.ff.net.0.proj.lora_B.weight",
        "transformer.transformer_blocks.3.ff.net.2.lora_A.weight",
        "transformer.transformer_blocks.3.ff.net.2.lora_B.weight",
        "transformer.transformer_blocks.3.attn.to_out.0.lora_A.weight",
        "transformer.transformer_blocks.3.attn.to_out.0.lora_B.weight",
        "transformer.token_refiner.refiner_blocks.1.attn.to_q.lora_A.weight",
        "transformer.token_refiner.refiner_blocks.1.attn.to_q.lora_B.weight",
        "transformer.time_embedder.linear_1.lora_A.weight",
        "transformer.time_embedder.linear_1.lora_B.weight",
        "transformer.endpoint_time_embedder.linear_2.lora_A.weight",
        "transformer.endpoint_time_embedder.linear_2.lora_B.weight",
    ]
    plan = plan_from_keys(keys)
    assert plan["blocks.3.mlp.fc1"]["kind"] == "lora"
    assert plan["blocks.3.mlp.fc2"]["kind"] == "lora"
    assert plan["blocks.3.attn.out_proj"]["kind"] == "lora"
    assert "token_refiner.blocks.1.attn.qkv_proj" in plan
    assert plan["time_embedder.proj_in"]["kind"] == "lora"
    assert plan["endpoint_time_embedder.proj_out"]["kind"] == "endpoint"


def test_fused_qkv_math():
    torch.manual_seed(0)
    r, inner, hidden = 4, 12, 8
    tensors = {}
    parts = {}
    for p in ("q", "k", "v"):
        a = torch.randn(r, hidden)
        b = torch.randn(inner, r)
        tensors[f"{p}.A"] = a
        tensors[f"{p}.B"] = b
        parts[p] = {"A": f"{p}.A", "B": f"{p}.B"}
    a3, b3 = fused_qkv_pair({"parts": parts}, tensors)
    assert a3.shape == (3 * r, hidden)
    assert b3.shape == (3 * inner, 3 * r)
    # block-diagonal B' times stacked A' equals the three per-part deltas stacked
    want = torch.cat([tensors[f"{p}.B"] @ tensors[f"{p}.A"] for p in ("q", "k", "v")], dim=0)
    assert torch.allclose(b3 @ a3, want, atol=1e-5)
    # q/k/v thirds land in the fused output order
    x = torch.randn(5, hidden)
    fused = torch.nn.functional.linear(torch.nn.functional.linear(x, a3), b3)
    per_part = torch.cat([torch.nn.functional.linear(
        torch.nn.functional.linear(x, tensors[f"{p}.A"]), tensors[f"{p}.B"])
        for p in ("q", "k", "v")], dim=-1)
    assert torch.allclose(fused, per_part, atol=1e-5)


@pytest.mark.parametrize("keys", [
    ["blocks.0.attn.out_proj.lora_A.weight"],                       # incomplete pair
    ["blocks.0.attn.out_proj.lora_A.weight",
     "blocks.0.attn.out_proj.lora_A.weight"],                       # duplicate
    ["blocks.0.attn.out_proj.lora_A.weight",
     "blocks.0.attn.out_proj.lora_B.weight",
     "blocks.0.attn.what.lora_A.weight",
     "blocks.0.attn.what.lora_B.weight"],                           # unknown module
    ["transformer.blocks.0.attn.to_q.lora_A.weight",
     "transformer.blocks.0.attn.to_q.lora_B.weight"],               # neither layout
])
def test_plan_errors(keys):
    with pytest.raises(ValueError):
        plan_from_keys(keys)


if __name__ == "__main__":
    test_converted_plan_kinds()
    test_original_plan_groups_qkv()
    test_original_plan_translates_names()
    test_fused_qkv_math()
    for keys in [
        ["blocks.0.attn.out_proj.lora_A.weight"],
        ["blocks.0.attn.out_proj.lora_A.weight", "blocks.0.attn.out_proj.lora_A.weight"],
        ["blocks.0.attn.out_proj.lora_A.weight", "blocks.0.attn.out_proj.lora_B.weight",
         "blocks.0.attn.what.lora_A.weight", "blocks.0.attn.what.lora_B.weight"],
        ["transformer.blocks.0.attn.to_q.lora_A.weight",
         "transformer.blocks.0.attn.to_q.lora_B.weight"],
    ]:
        try:
            plan_from_keys(keys)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {keys}")
    print("ALL PASS")
