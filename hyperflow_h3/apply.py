"""Applying the converted HyperFlow LoRA onto a cloned ModelPatcher.

Two application paths, both through ComfyUI's own weight-adapter machinery
(ported from the ComfyUI-MiniMax-H3-Turbo / VDN-H3 nodes, which battle-tested
them on this exact model family):

* bypass (default): run-time additive LoRA via BypassInjectionManager -- the
  base's own (possibly quantized) forward runs, the bf16 delta is added in
  activation space. Sharpest, costs extra peak VRAM.
* merge: fold backbone deltas into weights via add_patches; time projections
  remain in bypass to keep the endpoint branch free of the base time LoRA.

The fused attn.qkv_proj pairs (rank 3r, alpha 3x) use standard LoRA merge math;
bypass stores only the three nonzero blocks and evaluates separate branches. MLP fc2 modules whose base weight
rides ComfyUI's fused int8 matmul are invisible to bypass hooks and are routed
through merge in bypass mode (same detection the Turbo node uses).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter

_log = logging.getLogger("comfy.hyperflow")


class _FrugalLoRA(comfy.weight_adapter.LoRAAdapter):
    """LoRA bypass adapter with a memory-frugal additive path (ported from the
    MiniMax-H3-Turbo node): accumulates up(down(x)) * scale straight into the base
    output instead of allocating the full-size projection three times."""

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        if getattr(self, "is_conv", False):
            return super().bypass_forward(org_forward, x, *args, **kwargs)
        base_out = org_forward(x, *args, **kwargs)
        up, down, alpha = self.weights[0], self.weights[1], self.weights[2]
        rank = down.shape[0]
        scale = (alpha / rank if alpha is not None else 1.0) \
            * getattr(self, "multiplier", 1.0)
        down = down.to(dtype=x.dtype)
        up = up.to(dtype=x.dtype)
        return base_out.add_(F.linear(F.linear(x, down), up), alpha=scale)


class _SplitQKVLoRA(comfy.weight_adapter.LoRAAdapter):
    """Keep only the three nonzero blocks of the fused QKV LoRA."""

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        out = org_forward(x, *args, **kwargs)
        up, down, alpha = self.weights[:3]
        scale = (alpha / down.shape[0] if alpha is not None else 1.0) * self.multiplier
        width = up.shape[1]
        for i, (a, b) in enumerate(zip(down.chunk(3), up)):
            result = out[..., i * width:(i + 1) * width]
            result.add_(F.linear(F.linear(x, a.to(dtype=x.dtype)), b.to(dtype=x.dtype)), alpha=scale)
        return out


def _bypass_adapter(adapter, module):
    up, down = adapter.weights[:2]
    if module.endswith(".attn.qkv_proj") and up.shape[0] % 3 == 0 and down.shape[0] % 3 == 0:
        blocks = up.reshape(3, up.shape[0] // 3, 3, down.shape[0] // 3)
        if all(not torch.count_nonzero(blocks[i, :, j, :]).item()
               for i in range(3) for j in range(3) if i != j):
            compact = torch.stack([blocks[i, :, i, :] for i in range(3)])
            return _SplitQKVLoRA(adapter.loaded_keys, (compact, *adapter.weights[1:]))
    return _FrugalLoRA(adapter.loaded_keys, adapter.weights)


def _int8_fused_fc2(dm, modules):
    """MLP fc2 modules riding ComfyUI's fused int8 matmul: their fused forward
    reads linear.weight directly and never calls the module forward, so a bypass
    hook would silently drop the LoRA. Those must go through merge.
    (Ported from the MiniMax-H3-Turbo node.)"""
    fused = []
    for m in modules:
        if not m.endswith(".mlp.fc2"):
            continue
        try:
            w = comfy.utils.get_attr(dm, m + ".weight")
        except Exception:
            continue
        if (getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"
                and not getattr(getattr(w, "_params", None), "transposed", False)):
            fused.append(m)
    return fused


def _install_injection(new_model, hooks):
    """All bypass hooks go through ONE PatcherInjection whose eject unwinds in
    reverse (LIFO): forward-order eject restores a stale hook as module.forward
    and the next load captures that hook as its own "original" -- infinite
    self-recursion on reload. STACK-PROOFING: every clone shares ONE inner
    model, and ComfyUI ejects a clone's injections only when that clone is
    UNLOADED, so re-running the node would otherwise stack another full hook
    set on the same modules (2x, 3x the LoRA delta per rerun). The live hook
    set is tracked on the shared inner model and ejected before a new set goes
    in. (Ported from the VDN-H3 node.)"""
    if not hooks:
        return
    owner = new_model.model      # shared by every clone of this model

    def inject_all(model_patcher):
        old = getattr(owner, "_hyperflow_live_hooks", None)
        if old:
            for hook in reversed(old):
                hook.eject()
        for hook in hooks:
            hook.inject()
        owner._hyperflow_live_hooks = hooks

    def eject_all(model_patcher):
        for hook in reversed(hooks):
            hook.eject()
        if getattr(owner, "_hyperflow_live_hooks", None) is hooks:
            owner._hyperflow_live_hooks = None

    injection = comfy.patcher_extension.PatcherInjection(
        inject=inject_all, eject=eject_all)
    new_model.set_injections("hyperflow_lora", [injection])


def apply_lora(new_model, weights, strength: float, mode: str) -> str:
    """Apply the backbone LoRA (every weights.lora entry) onto ``new_model``.
    Validates that every target resolves on the loaded base and that shapes line
    up -- the file must match the base, errors are never warnings. Targets that
    exist only as state-dict entries (folded into a fused quantized op, so there
    is no module to hook) are routed through the merge path automatically."""
    model = new_model.model
    modules = sorted(weights.lora)
    sd_keys = set(model.state_dict().keys())
    # Two module trees: the BaseModel's own (keys "diffusion_model.blocks...") and
    # the inner diffusion model's via get_model_object (keys "blocks..."). Under
    # dynamic fast_disk loading patcher.model can behave like a proxy whose
    # named_modules drops the prefix -- try every combination.
    trees = [dict(model.named_modules())]
    try:
        dm = new_model.get_model_object("diffusion_model")
        trees.append(dict(dm.named_modules()))
    except Exception:
        dm = None

    def find_module(m):
        for tree in trees:
            for key in (m, "diffusion_model." + m):
                mod = tree.get(key)
                if mod is not None:
                    return mod
        return None

    def sd_has(m):
        return (m + ".weight") in sd_keys \
            or ("diffusion_model." + m + ".weight") in sd_keys

    no_module, missing, checked = set(), [], 0
    for m in modules:
        mod = find_module(m)
        if mod is None or not hasattr(mod, "forward"):
            if sd_has(m):
                no_module.add(m)            # quant-fused: no hookable module
            else:
                missing.append(m)
            continue
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor):
            continue                        # quantized layout: checked at cast time
        a, b = weights.lora[m]
        if a.shape[1] != w.shape[1] or b.shape[0] != w.shape[0]:
            raise RuntimeError(
                f"{weights.path.name}: {m} LoRA shapes A{tuple(a.shape)}/"
                f"B{tuple(b.shape)} vs model weight {tuple(w.shape)}; this file "
                "does not match the loaded base.")
        checked += 1
    if missing:
        raise RuntimeError(
            f"{weights.path.name}: {len(missing)} LoRA target(s) have no module and "
            f"no state-dict entry on this model, e.g. {missing[:4]}. The file does "
            "not match the loaded base; load the full MiniMax-H3 base this adapter "
            "was distilled on.")

    lora = {}
    for m in modules:
        a, b = weights.lora[m]
        lora[m + ".lora_A.weight"] = a.contiguous()
        lora[m + ".lora_B.weight"] = b.contiguous()
        lora[m + ".alpha"] = torch.tensor(weights.alphas[m])
    key_map = {m: "diffusion_model.{}.weight".format(m) for m in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)

    # invisible-to-hooks targets: fused-quant ops with no module (found via the
    # sd-key fallback above) plus int8 fc2 whose fused kernel bypasses the hook
    # even though the module exists (_int8_fused_fc2). Both go through merge.
    fc2_fused = set(no_module)
    if dm is not None:
        fc2_fused |= set(_int8_fused_fc2(dm, modules))
    if mode == "merge":
        # The endpoint embeds raw time weights plus its OWN LoRA. Keep the
        # base time LoRA in activation space so it cannot leak into that copy.
        fc2_fused |= {m for m in modules if not m.startswith("time_embedder.")}
    manager = comfy.weight_adapter.BypassInjectionManager()
    n = 0
    for key, adapter in loaded.items():
        m = key[len("diffusion_model."):-len(".weight")]
        if not sd_has(m):
            continue
        if m in fc2_fused:
            continue                        # merged below
        if isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            adapter = _bypass_adapter(adapter, m)
        elif not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            continue
        manager.add_adapter(key, adapter, strength=strength)
        n += 1
    manager.create_injections(model)
    _install_injection(new_model, manager.hooks)
    report = f"{n} bypass adapters ({len(manager.hooks)} injections)"
    if fc2_fused:
        sub = {key_map[m]: loaded[key_map[m]] for m in fc2_fused
               if key_map[m] in loaded}
        k = len(new_model.add_patches(sub, strength))
        report += (f", {k} weights merged ({checked} shape-checked)" if mode == "merge"
                   else f", {k} fused/int8 targets via merge")
    return report


__all__ = ["apply_lora"]
