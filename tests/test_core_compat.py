"""Integration checks for the ComfyUI September 20–21 loading changes."""
import copy
from types import SimpleNamespace

import pytest
import torch

import comfy.cli_args
import comfy.model_management
import comfy.model_prefetch
import comfy.sampler_helpers
from comfy.ldm.minimax.model import Attention, MiniMaxH3Model
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrapperExecutor, WrappersMP, get_all_wrappers
from hyperflow_h3.apply import apply_lora
from hyperflow_h3.compiler import install_compiler_workaround
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


@pytest.mark.parametrize("fail", [False, True])
def test_compiler_guard_encloses_native_minimax_outer_forward(fail, monkeypatch):
    monkeypatch.setattr(comfy.cli_args.args, "disable_comfy_compiler", False)
    monkeypatch.setattr(comfy.model_prefetch.comfy.memory_management, "aimdo_enabled", True)
    monkeypatch.setattr(comfy.model_management, "is_device_cuda", lambda device: True)
    patcher = ModelPatcher(torch.nn.Module(), torch.device("cpu"), torch.device("cpu"))
    install_compiler_workaround(patcher)
    comfy.sampler_helpers.prepare_model_patcher(patcher, {}, patcher.model_options)
    options = patcher.model_options["transformer_options"]

    def unexpected_graph(*args):
        raise AssertionError("HyperFlow entered the allocation compiler")

    monkeypatch.setattr(comfy.model_prefetch, "malloc_graph_begin", unexpected_graph)

    def forward(x, timestep, context, transformer_options, **kwargs):
        assert comfy.cli_args.args.disable_comfy_compiler
        if fail:
            raise RuntimeError("sampling failed")
        return x

    dm = SimpleNamespace(_forward=forward)
    x = [torch.ones(1), torch.ones(1)]
    executor = WrapperExecutor.new_executor(
        lambda: MiniMaxH3Model.forward(dm, x, torch.ones(1), None, options),
        get_all_wrappers(WrappersMP.APPLY_MODEL, options))
    if fail:
        with pytest.raises(RuntimeError, match="sampling failed"):
            executor.execute()
    else:
        assert executor.execute() is x
    assert not comfy.cli_args.args.disable_comfy_compiler
