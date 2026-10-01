"""Integration checks for the ComfyUI September 20–21 loading changes."""
import copy

import pytest
import torch

import comfy.model_management
from comfy.ldm.minimax.model import Attention
from comfy.model_patcher import ModelPatcher
from hyperflow_h3.apply import apply_lora
from test_apply import _LoRAWeights


@pytest.mark.parametrize("mode", ["merge", "bypass"])
@pytest.mark.parametrize("override", [False, True])
def test_native_attention_with_hyperflow_and_checkpoint_preference(mode, override, monkeypatch):
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    torch.manual_seed(21)
    attn = Attention(12, 2, 6, 1e-6, operations=torch.nn)
    reference = copy.deepcopy(attn)
    holder = torch.nn.Module()
    holder.diffusion_model = torch.nn.Module()
    block = torch.nn.Module()
    block.attn = attn
    holder.diffusion_model.blocks = torch.nn.ModuleList([block])
    patcher = ModelPatcher(holder, torch.device("cpu"), torch.device("cpu"))
    a = torch.randn(6, 12) * .05
    b = torch.block_diag(*(torch.randn(12, 2) * .05 for _ in range(3)))
    weights = _LoRAWeights({"blocks.0.attn.qkv_proj": (a, b)}, {"blocks.0.attn.qkv_proj": 6.})
    apply_lora(patcher, weights, 1., mode)
    with torch.no_grad():
        reference.qkv_proj.weight.add_(b @ a)
    x = torch.randn(5, 12)
    expected = reference(x)
    calls = []

    def preferred(*args, **kwargs):
        raise AssertionError("explicit attention override must take precedence")

    def attention_override(original, *args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    options = {}
    if override:
        attn.comfy_attention.function = preferred
        options["optimized_attention_override"] = attention_override
    for _ in range(2):
        try:
            patcher.patch_model(torch.device("cpu"))
            torch.testing.assert_close(attn(x, transformer_options=options), expected)
        finally:
            patcher.unpatch_model(torch.device("cpu"))
    assert len(calls) == (2 if override else 0)
