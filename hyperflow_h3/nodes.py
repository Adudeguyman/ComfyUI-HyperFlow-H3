"""Apply HyperFlow (8-step LoRA + two-time conditioning) to a MiniMax-H3 model."""

from __future__ import annotations

import logging

import folder_paths

from hyperflow_h3.apply import apply_lora
from hyperflow_h3.compiler import install_compiler_workaround
from hyperflow_h3.embedder import install_two_time
from hyperflow_h3.schedule import validate_sigmas, video_schedule_sigmas
from hyperflow_h3.weights import ensure_downloaded, load_weights, resolve_weights

_log = logging.getLogger("comfy.hyperflow")

_LAYOUT_HINT = (" Chain order: Load Diffusion Model -> ApplyHyperFlow -> "
                "(optional) Model Attention Backend / Model Sparse Attention -> "
                "SamplerCustomAdvanced with the SIGMAS output.")


def _check_base(model):
    """The adapter targets ComfyUI's native MiniMax-H3 diffusion model. Returns
    'full' or 'pruned' for the loaded base. Duck-typed on the inner module (the
    same way the VDN-H3 node detects it) -- the wrapper class varies."""
    try:
        dm = model.get_model_object("diffusion_model")
    except Exception:
        dm = None
    blocks = getattr(dm, "blocks", None)
    attn0 = getattr(blocks[0], "attn", None) if blocks else None
    if dm is None or attn0 is None or not hasattr(attn0, "qkv_proj"):
        raise RuntimeError(
            "ApplyHyperFlow needs a MiniMax-H3 MODEL (diffusion_model with "
            "blocks[].attn.qkv_proj). Connect a MiniMax-H3 load checkpoint / "
            "load diffusion model output first." + _LAYOUT_HINT)
    pruned = bool(getattr(dm, "use_adaln_curves", False)) \
        or not hasattr(dm, "time_embedder")
    return "pruned" if pruned else "full"


def _apply(model, hyperflow_file, strength, lora_mode, verbose,
           gate_override=None, sigmas_override=None,
           variant="auto", download_if_missing=False):
    base = _check_base(model)
    try:
        path = resolve_weights(hyperflow_file)
    except FileNotFoundError:
        if not download_if_missing:
            raise
        wanted = base if variant == "auto" else variant
        path = ensure_downloaded(wanted)
        _log.info("[hyperflow] downloaded %s (%s base) from %s", path.name,
                  wanted, "the configured Hugging Face repo")
    weights = load_weights(path)
    file_variant = "full" if weights.endpoint else "pruned"
    if base == "pruned" and file_variant == "full":
        raise RuntimeError(
            f"{weights.path.name} is the FULL-base build (time_embedder + endpoint "
            "LoRA) but the loaded base is a pruned/curve MiniMax-H3 with no "
            "time_embedder. Select the pruned-base build of the adapter, or load "
            "a full MiniMax-H3 base.")
    if base == "full" and file_variant == "pruned":
        raise RuntimeError(
            f"{weights.path.name} is the PRUNED-base build (backbone LoRA only) but "
            "the loaded base is a full MiniMax-H3. Two-time conditioning would stay "
            "off and the output would deviate from the released model. Select "
            "custom_node_hyperflow_8step_v1.0_comfyui.safetensors (the file without "
            "'_pruned') instead.")

    new_model = model.clone()
    report = apply_lora(new_model, weights, strength, lora_mode)
    gate = None
    if base == "full":
        gate = install_two_time(new_model, weights, gate_override, verbose)
    install_compiler_workaround(new_model)

    raw = None
    if sigmas_override:
        try:
            raw = [float(s) for s in sigmas_override.replace(",", " ").split()]
        except ValueError:
            raise ValueError(f"sigmas_override is not a comma/space separated list "
                             f"of numbers: {sigmas_override!r}")
        validate_sigmas(raw)
    sigmas = video_schedule_sigmas(weights.metadata.sigmas if raw is None else raw)
    _log.info(
        "[hyperflow] %s (%s build) applied on %s base [%s]: %s | %d LoRA modules, "
        "rank %d, gate %s, grid %d points",
        weights.path.name, variant, base, lora_mode, report,
        len(weights.lora), weights.rank,
        f"{gate:.4g}" if gate is not None else "n/a (backbone only)",
        sigmas.numel())
    return new_model, sigmas


def _file_combo():
    names = folder_paths.get_filename_list("hyperflow")
    return names or ["<download the converted HyperFlow .safetensors into models/hyperflow/>"]


class ApplyHyperFlow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {
                "tooltip": "The MiniMax-H3 diffusion model to patch. Chain once, "
                           "between the model loader and the sampler."}),
            "hyperflow_file": (_file_combo(), {
                "tooltip": "The ComfyUI-converted HyperFlow weights "
                           "(models/hyperflow). Get the converted build from "
                           "the HyperFlow Hugging Face repo: one .safetensors, "
                           "ComfyUI module paths, no on-the-fly conversion."}),
            "strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Adapter strength. 1.0 is the released model."}),
            "lora_mode": (["bypass", "merge"], {
                "default": "bypass",
                "tooltip": "bypass (default): LoRA applied at run time -- sharpest, "
                           "matches the reference's unmerged bf16 branches. merge: "
                           "folded into the weights -- lowest VRAM, softer on "
                           "quantized bases."}),
            "variant": (["auto", "full", "pruned"], {
                "default": "auto",
                "tooltip": "Which converted build to fetch when downloading: "
                           "auto (default) matches the detected base model; full "
                           "= non-pruned base (the released model); pruned = "
                           "backbone-only build for pruned/curve bases. Ignored "
                           "when the file is already on disk."}),
            "download_if_missing": ("BOOLEAN", {
                "default": False,
                "tooltip": "Download the chosen variant from the configured "
                           "Hugging Face repo (drbaph/Hyperflow-Comfyui) into "
                           "models/hyperflow when no weights file is there. "
                           "Fetches exactly the published .safetensors -- nothing "
                           "else. Requires internet."}),
            "verbose": ("BOOLEAN", {"default": False, "tooltip": "Log the applied "
                        "modules and the per-step (t, r) context."}),
        }}

    RETURN_TYPES = ("MODEL", "SIGMAS")
    RETURN_NAMES = ("model", "sigmas")
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = (
        "HyperFlow: 8-step LoRA + two-time (t, r) conditioning for MiniMax-H3. "
        "Apply between the model loader and the sampler, then feed the SIGMAS "
        "output into SamplerCustomAdvanced in place of a scheduler. Optionally "
        "stack the core Model Attention Backend and Model Sparse Attention nodes "
        "on top (sol-attn: start_percent 0.16, dense_blocks \"0,1\", tau 1.0).")

    def apply(self, model, hyperflow_file, strength, lora_mode, variant,
              download_if_missing, verbose):
        new_model, sigmas = _apply(model, hyperflow_file, strength, lora_mode,
                                   verbose, variant=variant,
                                   download_if_missing=download_if_missing)
        return (new_model, sigmas)


class ApplyHyperFlowAdvanced:
    """Everything the base node does, plus the ablation knobs: gate override and
    a custom sigma grid (both deviate from the released recipe)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {
                "tooltip": "The MiniMax-H3 diffusion model to patch. Chain once."}),
            "hyperflow_file": (_file_combo(), {
                "tooltip": "The ComfyUI-converted HyperFlow weights "
                           "(models/hyperflow)."}),
            "strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Adapter strength. 1.0 is the released model."}),
            "lora_mode": (["bypass", "merge"], {
                "default": "bypass",
                "tooltip": "bypass: run-time additive (sharp). merge: folded into "
                           "the weights (low VRAM, softer on quantized bases)."}),
            "variant": (["auto", "full", "pruned"], {
                "default": "auto",
                "tooltip": "Build to fetch when downloading: auto (default) matches "
                           "the detected base; full = non-pruned base; pruned = "
                           "backbone-only for pruned/curve bases. Ignored when the "
                           "file is already on disk."}),
            "download_if_missing": ("BOOLEAN", {
                "default": False,
                "tooltip": "Download the chosen variant from drbaph/"
                           "Hyperflow-Comfyui into models/hyperflow when missing. "
                           "Requires internet."}),
            "gate": ("FLOAT", {
                "default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05,
                "tooltip": "Blend of the endpoint embedding. -1 (default) reads the "
                           "gate from the weights file; any other value overrides "
                           "it (ablation)."}),
            "sigmas": ("STRING", {
                "default": "", "multiline": False,
                "tooltip": "Custom raw sigma grid, comma separated, must start <= 1 "
                           "and end at exactly 0 (e.g. the 9 default points). Empty "
                           "= the grid stored in the weights file (the trained "
                           "8-step grid). Ablation use only."}),
            "verbose": ("BOOLEAN", {"default": False, "tooltip": "Log the applied "
                        "modules and the per-step (t, r) context."}),
        }}

    RETURN_TYPES = ("MODEL", "SIGMAS")
    RETURN_NAMES = ("model", "sigmas")
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = (
        "HyperFlow advanced: gate and sigma-grid overrides for ablations. "
        "Defaults reproduce the released model exactly.")

    def apply(self, model, hyperflow_file, strength, lora_mode, variant,
              download_if_missing, gate, sigmas, verbose):
        new_model, sigmas_out = _apply(model, hyperflow_file, strength, lora_mode,
                                       verbose,
                                       gate_override=None if gate < 0 else gate,
                                       sigmas_override=sigmas or None,
                                       variant=variant,
                                       download_if_missing=download_if_missing)
        return (new_model, sigmas_out)


NODE_CLASS_MAPPINGS = {"ApplyHyperFlow": ApplyHyperFlow,
                       "ApplyHyperFlowAdvanced": ApplyHyperFlowAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyHyperFlow": "Apply HyperFlow (MiniMax-H3 8-Step)",
    "ApplyHyperFlowAdvanced": "Apply HyperFlow Advanced (Ablations)"}
