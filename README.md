# ComfyUI-Hyperflow

[中文文档 (README_ZH)](README_ZH.md)

<img width="1130" height="476" alt="image" src="https://github.com/user-attachments/assets/327cbbae-a2d7-4f86-b7ec-4dd12b6ce9e9" />

ComfyUI node pack for [HyperFlow](https://github.com/Video-Rebirth/hyperflow) — Video Rebirth's 8-step LoRA for MiniMax-H3 (video + audio), ported onto ComfyUI's native MiniMax-H3 model. No ComfyUI core files are modified; no custom sampler is needed.

> **Note — don't want this custom node?** Use the **standalone LoRA builds** instead — extracted and converted to plain ComfyUI format, they load with the **stock `Load LoRA` node**, nothing to install: [drbaph/MiniMax-H3-Turbo-Lora-ComfyUI](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/) — `minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors` (3.93 GB) / `..._pruned_bf16.safetensors` (3.91 GB), plus rank-20 resized variants (~318/316 MB). Those apply the **backbone LoRA only** — the two-time `(t, r)` conditioning is exclusive to this node, so without it the output deviates from the released model.

The adapter is two things on top of the official base: a LoRA (rank 256, unmerged bf16 branches) and **two-time `(t, r)` conditioning** — every step is conditioned on the interval it integrates, `r = 1 - sigma_next`. This pack blends the two embeddings and preserves distinct `(t, r)` rows through the native blocks and output heads, including the first step where video and audio share `t = 0`. All forward patches and LoRA injections use ComfyUI's ModelPatcher; no core files are changed.




https://github.com/user-attachments/assets/3e2e2d5c-6a39-46e9-8946-aa42f08c3d9d





## Install

1. Clone into `ComfyUI/custom_nodes/ComfyUI-Hyperflow`.
2. Download **one** converted weights file from the [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui) Hugging Face repo (or just enable the node's `download_if_missing` toggle and it fetches itself on first run):

```
📂 ComfyUI/
└── 📂 models/
    └── 📂 hyperflow/
        ├── custom_node_hyperflow_8step_v1.0_comfyui.safetensors         (3.67 GiB, full base — the released 8-step model)
        └── custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors  (3.64 GiB, pruned/curve bases — backbone only, single-time)
```

   The `hyperflow.json` manifests ship with this pack in `assets/` — users only ever download the `.safetensors`. Converted files carry the full self-describing HyperFlow header (gate, sigma grid, rank); the original diffusers-layout file is rejected with a clear message, never translated at load time.

3. Restart ComfyUI.

## Usage

```
Load Diffusion Model (MiniMax-H3)
  -> ApplyHyperFlow            (MODEL -> MODEL + SIGMAS)
  -> [optional] Model Attention Backend        (core node; dense fallback incl. comfy-kitchen int8)
  -> [optional] Model Sparse Attention         (core node; sol-attn / sla / vsa)
  -> SamplerCustomAdvanced + guider + Euler    (feed the SIGMAS output, not a scheduler)
```

- `ApplyHyperFlow` outputs the trained 9-point sigma grid as **SIGMAS** — wire it into `SamplerCustomAdvanced` in place of `BasicScheduler`. Use Euler for the released recipe; other samplers are ablations. The two-time endpoints are derived from `sample_sigmas`.
- `lora_mode`: `bypass` (default) applies the LoRA at run time — matches the reference's unmerged bf16 branches. `merge` folds the backbone LoRA into the weights; the two small base time projections stay in bypass so the endpoint branch reads unmodified weights. Merging quantized weights can change numerical results.
- `download_if_missing` fetches the chosen `variant` (`auto` matches the detected base) from [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui) into `models/hyperflow/` — exactly the published `.safetensors`, nothing else. Off by default.
- `ApplyHyperFlowAdvanced` adds gate and sigma-grid overrides (ablations; defaults reproduce the released model).

### Sol-Attn (optional sparse attention) — core node settings

HyperFlow's validated Sol-Attn recipe maps onto the core **Model Sparse Attention** node:

| HyperFlow (reference) | Model Sparse Attention |
| --- | --- |
| `dense_steps = 2` (of 8) | `start_percent = 0.16` |
| `dense_layers = (0, 1)` | `dense_blocks = "0,1"` |
| `tau = 1.0` | `tau = 1.0` (method `sol-attn`) |
| sink tokens: none | `sink_conditioning = "off"` |
| — | `extra_tokens = 0` |

`start_percent` is measured on the unshifted noise clock, not as a fraction of steps. With the shipped grid and shifts, 0.16 keeps exactly the first two evaluations dense. Sparse attention itself is an upstream-supported approximation; use dense attention for the dense recipe.

SLA is a different sparse method — use it only with SLA-trained weights.

### Pruned / curve bases — experimental curve refit

Pruned MiniMax-H3 bases have no `time_embedder`, so the two-time `(t, r)` pathway cannot apply — by default the pruned build applies the backbone LoRA only and runs single-time (off-recipe). The **experimental curve refit** restores most of the two-time signal through a tiny checkpoint-bound fit (~34 KB, bundled in `assets/curve_fits/` for the FL2VA and REF2VA pruned int8 checkpoints — no download, no extra model weights).



https://github.com/user-attachments/assets/e72c45e3-2cc3-48ce-bb1a-2f0c25e7d602



**Recommended settings for pruned:**

- Enable **`experimental_curve_refit`** (widget on both Apply nodes; right-click → *show optional widgets* if hidden), keep **`strength = 1.0`**, and leave gate/sigma overrides empty. Matching is tiered: the exact fitted files apply silently; byte-different copies of the fitted base or adapter (mirrors, HF downloads) still apply **best-effort with a console warning** — the recipe (strength/gate/sigmas) is always validated; unknown checkpoints or a modified MODEL fall back to backbone-only, with the file hashes in the log for support.
- Use **`bypass`** as `lora_mode` for the sharpest match to the reference branches.
- Same sampling as full bases: **Euler** + the node's **SIGMAS** output + the Sol-Attn table above.
- Console: `[hyperflow] curve refit disabled: <reason>` means the fallback engaged — check strength/overrides/checkpoint.

Measured against the full base (fixed prompt/seed/latents): FL2VA video cosine **0.81 → 0.88**, REF2VA audio cosine **0.93 → 0.99**. It recovers *most* — not all — of the missing conditioning; the full base remains the reference. Details and per-family numbers: [`validation/curve_findings.md`](validation/curve_findings.md). New pruned checkpoints need their own fit (`validation/curve_spike.py`).

## Notes

- Existing node weights are corrected on load: the loader swaps legacy Diffusers SwiGLU rows into ComfyUI order. No replacement download is needed. New conversions carry `hyperflow_fc1_layout=gate_value` and are not swapped again.
- Bypass QKV adapters store the three separate LoRA projections instead of the block-diagonal zero padding, saving 1,092 MiB of adapter tensor storage for the released rank-256 build.

- **Base detection is automatic**: the node inspects the loaded model — full base (has `time_embedder`) or pruned/curve base (no `time_embedder`) — and enforces the matching weights build with a clear error that names the right file. The pruned-base build applies the backbone LoRA only and runs single-time (off-recipe).
- **Experimental curve refit** for pruned/curve bases: see the dedicated section above.
- **Quantized bases** (int8/fused ops): LoRA targets that the base folds into a fused kernel (no hookable module) are detected and applied through the merge path automatically — the console report lists them as `N fused/int8 targets via merge`.
- **Model sampling shifts**: the H3 model already defaults to video shift 12 / audio shift 3. The core ModelSampling node goes after `ApplyHyperFlow` in the chain, and is only needed if you want non-default shifts.
- **aimdo malloc-graph**: on Comfy builds whose model compiler crashes on patched MiniMax-H3 forwards, the node disables the compiler for its own model calls.
- Weights are a Model Derivative of MiniMax-H3 under the [MiniMax H3 Community License](https://huggingface.co/videorebirth/hyperflow); this pack's code is Apache-2.0 (the schedule/embedder ports derive from the HyperFlow and diffusers code, see the upstream `THIRD_PARTY_NOTICES.md`).

## Stuck at “Model Initializing” after a ComfyUI update

That status covers the whole first sampling step, including deferred weight loading and LoRA merging. It does not identify a compiler hang. `merge` reduces resident adapter memory, but still needs temporary memory and time to patch the base weights.

ComfyUI v0.37 introduced automatic fast-disk detection compared with v0.36. To isolate a stall after updating, restart ComfyUI with `--disable-fast-disk` and retry the same workflow. Treat this as a diagnostic, not a confirmed HyperFlow fix. If it still stalls, report the ComfyUI commit, full console log, base/LoRA filenames, GPU/VRAM and system RAM, and whether it happens on the first chunk or a later one. Include the `[hyperflow] ... applied` line if present.

## Acknowledgements

- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) (GitHub): the base model, VAEs, conditioner and official workflows.
- [AnyFlow](https://github.com/NVlabs/AnyFlow) (GitHub; Gu et al., 2026): the flow-map formulation behind the two-time (t, r) conditioning.
- [Sol-Attn](https://github.com/NVlabs/Sol-Attn) (GitHub; Li et al., 2026): optional attention kernels.

Thanks to their authors.
