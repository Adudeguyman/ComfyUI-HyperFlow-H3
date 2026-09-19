# ComfyUI-Hyperflow

[中文文档 (README_ZH)](README_ZH.md)

<img width="1475" height="663" alt="image" src="https://github.com/user-attachments/assets/1a476e02-d108-4184-b850-994c8094a32b" />


ComfyUI node pack for [HyperFlow](https://github.com/Video-Rebirth/hyperflow) — Video Rebirth's 8-step LoRA for MiniMax-H3 (video + audio), ported onto ComfyUI's native MiniMax-H3 model. No ComfyUI core files are modified; no custom sampler is needed.

> **Note — don't want this custom node?** Use the **standalone LoRA builds** instead — extracted and converted to plain ComfyUI format, they load with the **stock `Load LoRA` node**, nothing to install: [drbaph/MiniMax-H3-Turbo-Lora-ComfyUI](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/) — `minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors` (3.93 GB) / `..._pruned_bf16.safetensors` (3.91 GB), plus rank-20 resized variants (~318/316 MB). Those apply the **backbone LoRA only** — the two-time `(t, r)` conditioning is exclusive to this node, so without it the output deviates from the released model.

The adapter is two things on top of the official base: a LoRA (rank 256, unmerged bf16 branches) and **two-time `(t, r)` conditioning** — every step is conditioned on the interval it integrates, `r = 1 - sigma_next`. This pack replaces exactly one call in the native model (`time_embedder(t_vals)`) with the blended embedding and applies the LoRA through ComfyUI's own weight-adapter machinery (bypass default / merge optional), so quantized bases, model offloading, and the memory visualizer all behave exactly like any other MODEL patch.



https://github.com/user-attachments/assets/86f10b24-8ac1-4290-8e5e-ebcb3c274f93



https://github.com/user-attachments/assets/12724013-821d-45d4-9bcf-eeecb33aa471





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

- `ApplyHyperFlow` outputs the trained 9-point sigma grid as **SIGMAS** — wire it into `SamplerCustomAdvanced` in place of `BasicScheduler`. Any core sampler/guider works; the two-time endpoints are derived from `sample_sigmas` (the same mechanism the native H3 final layer uses).
- `lora_mode`: `bypass` (default) applies the LoRA at run time — matches the reference's unmerged bf16 branches. `merge` folds it into the weights — lowest VRAM, softer on quantized bases.
- `download_if_missing` fetches the chosen `variant` (`auto` matches the detected base) from [drbaph/Hyperflow-Comfyui](https://huggingface.co/drbaph/Hyperflow-Comfyui) into `models/hyperflow/` — exactly the published `.safetensors`, nothing else. Off by default.
- `ApplyHyperFlowAdvanced` adds gate and sigma-grid overrides (ablations; defaults reproduce the released model).

### Sol-Attn (optional sparse attention) — core node settings

HyperFlow's validated Sol-Attn recipe maps onto the core **Model Sparse Attention** node:

| HyperFlow (reference) | Model Sparse Attention |
| --- | --- |
| `dense_steps = 2` (of 8) | `start_percent = 0.25` |
| `dense_layers = (0, 1)` | `dense_blocks = "0,1"` |
| `tau = 1.0` | `tau = 1.0` (method `sol-attn`) |
| sink tokens: none | keep default `exact_kv_and_rows` (strictly more exact on cond rows) |
| — | `extra_tokens = 0` for the closest recipe match (256 = more quality headroom) |

SLA is a different sparse method — use it only with SLA-trained weights.

## Notes

- **Base detection is automatic**: the node inspects the loaded model — full base (has `time_embedder`) or pruned/curve base (no `time_embedder`) — and enforces the matching weights build with a clear error that names the right file. The pruned-base build applies the backbone LoRA only and runs single-time (off-recipe).
- **Quantized bases** (int8/fused ops): LoRA targets that the base folds into a fused kernel (no hookable module) are detected and applied through the merge path automatically — the console report lists them as `N fused/int8 targets via merge`.
- **Model sampling shifts**: the H3 model already defaults to video shift 12 / audio shift 3. The core ModelSampling node goes after `ApplyHyperFlow` in the chain, and is only needed if you want non-default shifts.
- **aimdo malloc-graph**: on Comfy builds whose model compiler crashes on patched MiniMax-H3 forwards, the node disables the compiler for its own model calls.
- Weights are a Model Derivative of MiniMax-H3 under the [MiniMax H3 Community License](https://huggingface.co/videorebirth/hyperflow); this pack's code is Apache-2.0 (the schedule/embedder ports derive from the HyperFlow and diffusers code, see the upstream `THIRD_PARTY_NOTICES.md`).

## Acknowledgements

- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) (GitHub): the base model, VAEs, conditioner and official workflows.
- [AnyFlow](https://github.com/NVlabs/AnyFlow) (GitHub; Gu et al., 2026): the flow-map formulation behind the two-time (t, r) conditioning.
- [Sol-Attn](https://github.com/NVlabs/Sol-Attn) (GitHub; Li et al., 2026): optional attention kernels.

Thanks to their authors.
