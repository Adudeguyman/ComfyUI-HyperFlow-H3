"""Bypass/merge application tests: adapter math, hook cycles, LIFO injection."""
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_COMFYUI_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_COMFYUI_ROOT))
sys.path.insert(0, str(_PACKAGE))

import comfy.lora
import comfy.model_management
from comfy.weight_adapter.bypass import BypassForwardHook

from hyperflow_h3.apply import _FrugalLoRA, _install_injection

comfy.model_management.get_torch_device = lambda: torch.device("cpu")


def _adapter(up=None, down=None, alpha=4.0):
    torch.manual_seed(0)
    up = up or torch.randn(8, 4)
    down = down or torch.randn(4, 8) * 0.1
    return _FrugalLoRA(set(), (up, down, alpha, None, None, None))


def test_frugal_bypass_math():
    torch.manual_seed(1)
    mod = nn.Linear(8, 8)
    orig_forward = mod.forward
    x = torch.randn(3, 8)
    adapter = _adapter()
    hook = BypassForwardHook(mod, adapter, multiplier=0.7)
    hook.inject()
    try:
        got = mod(x)
    finally:
        hook.eject()
    up, down, alpha = adapter.weights[0], adapter.weights[1], adapter.weights[2]
    base = torch.nn.functional.linear(x, mod.weight, mod.bias)
    want = base + (alpha / down.shape[0]) * 0.7 * torch.nn.functional.linear(
        torch.nn.functional.linear(x, down), up)
    assert torch.allclose(got, want, atol=1e-5)
    assert mod.forward == orig_forward


def test_merge_calculate_weight():
    torch.manual_seed(2)
    w = torch.randn(8, 8)
    a = torch.randn(4, 8) * 0.1     # lora_A [rank, in]
    b = torch.randn(8, 4) * 0.5     # lora_B [out, rank]
    lora = {"m.lora_A.weight": a, "m.lora_B.weight": b, "m.alpha": torch.tensor(4.0)}
    loaded = comfy.lora.load_lora(lora, {"m": "diffusion_model.m.weight"},
                                  log_missing=False)
    adapter = loaded["diffusion_model.m.weight"]
    merged = adapter.calculate_weight(w.clone(), "diffusion_model.m.weight",
                                      1.0, 1.0, None, lambda t: t)
    want = w + (4.0 / 4) * (b @ a)
    assert torch.allclose(merged, want, atol=1e-5)


def test_load_lora_key_map_flow():
    """The exact dict/key_map flow apply.py uses, end to end on a module tree."""
    torch.manual_seed(3)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = nn.Linear(8, 24, bias=False)

    holder = nn.Module()
    holder.diffusion_model = M()
    a3 = torch.randn(12, 8) * 0.1
    b3 = torch.randn(24, 12) * 0.1
    lora = {"diffusion_model.qkv_proj.lora_A.weight": a3,
            "diffusion_model.qkv_proj.lora_B.weight": b3,
            "diffusion_model.qkv_proj.alpha": torch.tensor(12.0)}
    loaded = comfy.lora.load_lora(lora, {"diffusion_model.qkv_proj":
                                         "diffusion_model.diffusion_model.qkv_proj.weight"},
                                  log_missing=False)
    adapter = loaded["diffusion_model.diffusion_model.qkv_proj.weight"]
    assert isinstance(adapter, comfy.weight_adapter.LoRAAdapter)
    w = holder.diffusion_model.qkv_proj.weight
    merged = adapter.calculate_weight(w.clone(), "k", 1.0, 1.0, None, lambda t: t)
    assert torch.allclose(merged, w + (12.0 / 12) * (b3 @ a3), atol=1e-5)


class _Patcher:
    def __init__(self):
        self.injections = {}
        self.model = types.SimpleNamespace()

    def set_injections(self, key, value):
        self.injections[key] = value


def _tiny_dm(with_fc2=True):
    """A tiny stand-in for the MiniMaxH3Model tree (blocks[].attn.qkv_proj etc.)."""
    torch.manual_seed(7)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = nn.Linear(8, 24, bias=False)
            self.out_proj = nn.Linear(8, 8, bias=False)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16, bias=False)
            if with_fc2:
                self.fc2 = nn.Linear(16, 8, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()
            self.mlp = MLP()

    class DM(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(2)])

    return DM()


class _ProxyModel:
    """Mimics this ComfyUI's fast_disk patcher.model: getattr resolves through
    'diffusion_model.<path>' (injection walks work), but named_modules() and
    state_dict() return BARE inner keys (no 'diffusion_model.' prefix)."""

    def __init__(self, inner, extra_sd=None):
        self._inner = inner
        self._extra_sd = extra_sd or {}

    def __getattr__(self, name):
        if name == "diffusion_model":
            return self._inner
        return getattr(self._inner, name)

    def named_modules(self):
        return self._inner.named_modules()

    def state_dict(self):
        sd = dict(self._inner.state_dict())
        sd.update(self._extra_sd)
        return sd


class _LoRAWeights:
    def __init__(self, lora, alphas):
        self.lora = lora
        self.alphas = alphas
        self.endpoint = {}
        self.path = Path("fake.safetensors")


def _weights_for(dm):
    r = 4
    lora, alphas = {}, {}
    for name, w in dm.state_dict().items():
        if not name.endswith(".weight"):
            continue
        mod = name[:-len(".weight")]
        a = torch.randn(r, w.shape[1]) * 0.05
        b = torch.randn(w.shape[0], r) * 0.05
        lora[mod] = (a, b)
        alphas[mod] = float(r)
    return _LoRAWeights(lora, alphas)


class _ApplyPatcher:
    def __init__(self, inner, extra_sd=None):
        self.model = _ProxyModel(inner, extra_sd)
        self._inner = inner
        self.patches = {}
        self.injections = {}

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self._inner

    def add_patches(self, patches, strength):
        self.patches.update(patches)
        return list(patches)

    def set_injections(self, key, value):
        self.injections[key] = value


def test_apply_resolves_proxy_model_merge():
    """Regression: named_modules/state_dict without the diffusion_model prefix
    (fast_disk proxy) must not be reported missing."""
    dm = _tiny_dm()
    patcher = _ApplyPatcher(dm)
    report = __import__("hyperflow_h3.apply", fromlist=["apply_lora"]).apply_lora(
        patcher, _weights_for(dm), 1.0, "merge")
    assert f"{len(dm.state_dict())} weights merged" in report


def test_apply_resolves_proxy_model_bypass():
    dm = _tiny_dm()
    patcher = _ApplyPatcher(dm)
    from hyperflow_h3.apply import apply_lora
    report = apply_lora(patcher, _weights_for(dm), 1.0, "bypass")
    assert "bypass adapters" in report
    assert "hyperflow_lora" in patcher.injections


def test_apply_fc2_without_module_routes_to_merge():
    """A target present in the state dict but folded away as a module (fused
    quant op) must go through merge, not raise."""
    dm = _tiny_dm(with_fc2=False)
    extra = {"blocks.0.mlp.fc2.weight": torch.randn(8, 16)}
    patcher = _ApplyPatcher(dm, extra_sd=extra)
    full = _weights_for(dm)
    r = 4
    full.lora["blocks.0.mlp.fc2"] = (torch.randn(r, 16) * 0.05,
                                     torch.randn(8, r) * 0.05)
    full.alphas["blocks.0.mlp.fc2"] = float(r)
    from hyperflow_h3.apply import apply_lora
    report = apply_lora(patcher, full, 1.0, "bypass")
    assert "fused/int8 targets via merge" in report
    assert "diffusion_model.blocks.0.mlp.fc2.weight" in patcher.patches


def test_lifo_injection_cycles():
    """3x inject/eject keeps the true forward and correct values (the VDN
    regression: forward-order eject self-recurses on reload)."""
    torch.manual_seed(4)
    mod = nn.Linear(8, 8)
    true_fwd = mod.forward
    hooks = [BypassForwardHook(mod, _adapter(), multiplier=1.0) for _ in range(2)]
    patcher = _Patcher()
    _install_injection(patcher, hooks)
    injection = patcher.injections["hyperflow_lora"][0]
    x = torch.randn(3, 8)
    base = torch.nn.functional.linear(x, mod.weight, mod.bias)

    def delta(hook):
        up, down, alpha = hook.adapter.weights[0], hook.adapter.weights[1], hook.adapter.weights[2]
        return (alpha / down.shape[0]) * torch.nn.functional.linear(
            torch.nn.functional.linear(x, down), up)

    want = base + delta(hooks[0]) + delta(hooks[1])
    for cycle in range(3):
        injection.inject(patcher)
        assert mod.forward == hooks[1]._bypass_forward
        assert torch.allclose(mod(x), want, atol=1e-5)
        injection.eject(patcher)
        assert mod.forward == true_fwd
        assert hooks[0].original_forward is None and hooks[1].original_forward is None


if __name__ == "__main__":
    test_frugal_bypass_math()
    test_merge_calculate_weight()
    test_load_lora_key_map_flow()
    test_lifo_injection_cycles()
    print("ALL PASS")
