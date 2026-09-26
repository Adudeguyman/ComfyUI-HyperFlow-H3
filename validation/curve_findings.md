# HyperFlow on curve bases: conditioning spike + refit gate

2026-09-22. Branch `exp-curve`, based on `main` at `b4bd9cf`.
Phase 2 (integration + A/B/C gate) completed same day; results at the end.

**Recommendation: pursue a teacher-fitted table indexed by the actual `(t, r)`
pair, with separate video/audio rows (a revision of Option 2). Do not enable
Option 1 as the default based on this spike.** Coordinate blending can route
two different endpoints, but it does not adequately reproduce the learned
conditioning change. This is a numerical finding, not a visual-quality verdict.

The investigation stops before production integration, following the requested
stop condition rather than presenting coordinate blending as a working port.
No `curve.py` runtime patch was installed. The new files are validation only.

## Open questions

1. Both local FL2VA and REF2VA pruned checkpoints contain an fp32
   `adaln_t_table[1025, 8]`. Detection sets `time_embed_dim=8` from that table.
   All 50 blocks have `adaln_proj.linear.weight[96768, 8]` (stored fp16,
   configured fp32 at runtime). Their outputs reshape to three modalities,
   six modulation groups, and 5376 channels. The groups are attention
   shift/scale/gate and MLP shift/scale/gate. The final layer also consumes
   the curve: `[10752, 8]`, giving shift/scale for 5376 channels. The token
   refiner takes text and transformer options, with **no timestep input**.

2. Curves use exactly the same sigma computation as full bases. Native
   `_forward` computes video time from `timestep / 1000` and audio time via
   `time_shift_sigma`, respecting the transformer-option shift overrides.
   The payload supplies layout, augmentation and conditioning; it does not
   provide a ready-made `t_vals` vector.

3. The endpoint LoRA is small in absolute embedding units, but not negligible
   relative to the embedding or to HyperFlow's change. Statistics below cover
   the 16 endpoint evaluations of the actual video/audio eight-step schedule,
   with duplicate `r=1` included. Full rank-256 node-build weights, gate 0.25,
   strength 1 were used.

   | Base | Comparison at r | Relative L2 range | Minimum cosine | Maximum absolute difference |
   |---|---|---:|---:|---:|
   | FL2VA | endpoint vs raw base | 4.93–13.30% | 0.991121 | 0.023879 |
   | FL2VA | endpoint vs adapted base | 1.65–6.73% | 0.997759 | 0.028113 |
   | REF2VA | endpoint vs raw base | 4.42–12.97% | 0.991560 | 0.020111 |
   | REF2VA | endpoint vs adapted base | 1.69–7.31% | 0.997370 | 0.029998 |

   A separate FL2VA check at `t=0` gives endpoint-vs-raw relative L2 13.48%,
   cosine 0.990888, max absolute difference 0.009442. The main time-embedder
   LoRA also matters: adapted-vs-raw relative L2 is 11.98% at zero. The
   pruned file drops **both** learned branches.

4. Native code builds `unique_t`, then a vector `t_vals`, with segment indices
   and per-token tensor indices for mixed masks. References/conditions use
   `max(current_time, payload_augmentation)`. Fully masked target tokens use
   the model's standard visual/audio pin constants; partially masked tokens
   can introduce additional times. Text follows the video clock even when
   its modality tags vary. The final heads use the same scalar/tensor row
   indexing. Therefore per-pair expansion and row remapping are essential:
   changing one shared `t=0` entry cannot express two endpoints.

Source inspected: local `comfy/ldm/minimax/model.py`, especially `AdalnProj`,
`TokenRefiner`, `FinalLayer`, and `_forward` lines 625–780; detection in
`comfy/model_detection.py`. No core files were changed.

## Why coordinate interpolation is insufficient

The full target is

```
m_b(t,r) = W_b * silu(0.75 * E_main(t) + 0.25 * E_endpoint(r)) + bias_b
```

The pruned table approximates the **raw base's post-SiLU curve**, not either
adapted embedder. Option 1 uses

```
m'_b(t,r) = C_b * table(0.75*t + 0.25*r) + curve_bias_b
```

These differ through the two missing LoRAs, the nonlinear embedding function,
and SiLU after the blend. Piecewise linear lookup does not make interpolation
across many table intervals commute with blending two embeddings. High cosine
similarity between embeddings is insufficient evidence of equivalent modulation.

The table faithfully reproduces the *unadapted* pathway: mean relative
modulation error is only 0.197% for FL2VA and 0.193% for REF2VA. The much larger
HyperFlow error is therefore not simply the ordinary pruning approximation.

## Spike results

The script streams all 50 full AdaLN matrices plus the final AdaLN projection,
using Comfy Kitchen's own INT8/convrot dequantizer. It loads the conditioning
weights and both time LoRAs, not the entire transformer. Computation uses fp32
projections to isolate conditioning math; it does not simulate runtime bf16
embedding rounding or dynamic activation quantization. This is a comparison
against the locally available INT8 full checkpoint, not an original bf16 teacher.

Aggregate values below are arithmetic means of relative L2 errors, equally
weighted across 50 blocks × 16 time pairs × 3 modalities × 6 modulation groups.
They include modality/time combinations not selected during a normal forward.
Final-layer errors are stored separately in the JSON reports. These are not
percentages of image or audio quality.

| Method | FL2VA error | REF2VA error |
|---|---:|---:|
| Current backbone-only | 2.3658% | 3.3447% |
| Option 1: blended time coordinate | 2.4348% | 3.1855% |
| Blend the two curve table outputs | 2.2777% | 3.0275% |
| Joint unconstrained eight-coordinate refit | **1.2014%** | **1.2053%** |

Relative to the actual HyperFlow modulation *change* (target minus raw full
base), Option 1's average error is **115.85% / 104.20%** for FL2VA / REF2VA.
The refit reduces that to **46.82% / 31.69%**. It is a substantial improvement,
but still does not reproduce the learned pathway exactly. The refit minimizes
total squared error, not the mean relative-error statistic shown here.

Per-step errors below average the six groups over all 50 blocks, selecting the
actual video modality at video times and audio modality at audio times. Detailed
per-step/per-modality/per-group max-absolute, relative-L2 and cosine statistics
are in `curve_fl2va.json` and `curve_ref2va.json`.

| Base / stream / step | Backbone | Coordinate | Table blend | Refit |
|---|---:|---:|---:|---:|
| FL2VA video 1 | 2.02% | 2.04% | 2.04% | 1.09% |
| FL2VA video 2 | 2.03% | 2.03% | 2.03% | 1.09% |
| FL2VA video 3 | 2.08% | 2.03% | 2.03% | 1.08% |
| FL2VA video 4 | 2.23% | 2.08% | 2.02% | 1.06% |
| FL2VA video 5 | 2.08% | 2.12% | 1.97% | 1.03% |
| FL2VA video 6 | 1.75% | 1.96% | 1.90% | 1.01% |
| FL2VA video 7 | 1.61% | 1.89% | 1.87% | 1.00% |
| FL2VA video 8 | 3.74% | 2.89% | 2.10% | 1.00% |
| FL2VA audio 1 | 3.17% | 3.24% | 3.22% | 1.63% |
| FL2VA audio 2 | 3.12% | 3.24% | 3.19% | 1.61% |
| FL2VA audio 3 | 2.96% | 3.21% | 3.12% | 1.57% |
| FL2VA audio 4 | 2.68% | 3.12% | 3.06% | 1.54% |
| FL2VA audio 5 | 2.31% | 2.97% | 3.00% | 1.52% |
| FL2VA audio 6 | 2.08% | 2.77% | 2.96% | 1.50% |
| FL2VA audio 7 | 2.21% | 2.84% | 2.94% | 1.50% |
| FL2VA audio 8 | 4.32% | 4.49% | 2.84% | 1.46% |
| REF2VA video 1 | 3.06% | 3.11% | 3.11% | 1.09% |
| REF2VA video 2 | 3.02% | 3.09% | 3.09% | 1.08% |
| REF2VA video 3 | 2.97% | 3.08% | 3.07% | 1.07% |
| REF2VA video 4 | 2.96% | 3.13% | 3.04% | 1.06% |
| REF2VA video 5 | 2.90% | 3.14% | 2.95% | 1.04% |
| REF2VA video 6 | 2.87% | 2.94% | 2.85% | 1.01% |
| REF2VA video 7 | 3.12% | 2.95% | 2.81% | 1.00% |
| REF2VA video 8 | 5.46% | 3.77% | 3.16% | 1.00% |
| REF2VA audio 1 | 3.38% | 3.43% | 3.43% | 1.64% |
| REF2VA audio 2 | 3.26% | 3.41% | 3.36% | 1.62% |
| REF2VA audio 3 | 3.17% | 3.39% | 3.26% | 1.59% |
| REF2VA audio 4 | 3.20% | 3.30% | 3.16% | 1.56% |
| REF2VA audio 5 | 3.36% | 3.17% | 3.08% | 1.53% |
| REF2VA audio 6 | 3.65% | 3.06% | 3.02% | 1.51% |
| REF2VA audio 7 | 3.99% | 2.98% | 2.99% | 1.50% |
| REF2VA audio 8 | 5.00% | 4.18% | 3.03% | 1.44% |

First step explicitly: `t_v=t_a=0`, `r_v=0.006090224`, `r_a=0.023923874`.
At gate 0.25, the blended coordinates are **0.001522556** and **0.005980968**
(table indices approximately 1.55910 and 6.12451). They are distinct, so the
collision is solvable. Distinct coordinates alone do not ensure correct values:
FL2VA audio modulation error rises from 3.17% to 3.24% on that first step.

## Options and proposed next implementation

- **Option 1:** no weights, easy distribution, but the current numeric result
  does not justify enabling it automatically. Audio on FL2VA worsens at all
  eight steps. Neither this nor table-output blending recovers the learned
  time-LoRA changes.
- **Revised Option 2:** fit a shared eight-vector for each actual `(t,r)` pair
  jointly against all block projections and the final projection. The spike
  solves the normal equations in float64, including biases and the full
  post-blend SiLU target. Generated rows cost only `16*8*4 = 512` bytes per
  base before metadata. Index by pair/stream and step, not by `t` alone. Reuse
  the existing row-map routing for blocks and final heads. This removes the
  alleged need to squeeze audio/video collisions through a single scalar
  coordinate. The fit still has residual error outside the existing basis.
- **Option 3:** restoring approximately 13B weights costs roughly 26 GB at
  bf16 and defeats pruning; not pursued.
- **Option 4:** a curve-native endpoint adapter remains a longer-term design:
  freeze the curve backbone, train a small `(t,r)` module into table space
  plus low-rank per-block/final modulation residuals, and distill against the
  full HyperFlow teacher. Train video, audio, text, references, augmented
  conditions and masked rows, including the `t=0` collision. Use both
  modulation and denoising-output losses, then assess generated video/audio.
  The [upstream repository](https://github.com/Video-Rebirth/hyperflow) exists,
  but its README explicitly describes a released loader and examples, with
  only the LoRA released. A usable training pipeline was **not established**;
  do not assume upstream training code is available. No training was run.

Before shipping Option 2, fit and validate pinned `(t,t)` rows too: these still
mix two different learned embedders in the full model. A sampled pinned curve
of 1025 eight-vectors adds about 32 KiB. Arbitrary masks, augmentation, custom
sigmas, gate and LoRA strength need explicit handling; this 16-row spike only
tests generated rows at the default recipe. Bind any distributed fit to its
base/adapter/recipe, and compare FL2VA and REF2VA separately. Do not claim
that one table works for every pruned checkpoint or every LoRA strength.

If the remaining eight-dimensional fit error proves visible, an alternative
without training is to export per-block modulation residuals for the supported
pairs. Sixteen full generated-pair rows across the 50 blocks and final layer
cost about 148 MiB in fp16, before pinned/masked support; a factored residual
may be smaller. That directly represents directions missing from the curve
basis and is still far below a 13B-parameter transplant. This is a design
estimate, not a validated artifact.

## Validation status and reproduction

- Python: requested `C:\Users\drbaph\Documents\ComfyUI\venv\Scripts\python.exe`.
- GPU: RTX 5090, used for the numerical spike.
- Regression baseline: **63 passed** (the 57 tracked tests plus six cases in
  the pre-existing untracked `test_core_compat.py`).
- Full-base runtime files: unchanged from `main`; no production behavior
  change to compare. No separate bit-exact generation comparison was run.
- No curve runtime tests or A/B generation were run because the proposed
  coordinate implementation was stopped at the numerical validation stage.
  GPU availability is not the blocker. No visual or latent similarity claim
  is made. The refit remains a candidate, pending routing/pin tests and A/B/C
  generation using fixed prompt, seed, initial latents and sampler settings.
- Existing README edits and the untracked core-compatibility test were
  preserved. Nothing committed or pushed.

Run `curve_spike.py` from the node-pack directory with the ComfyUI venv:

```powershell
& C:\Users\drbaph\Documents\ComfyUI\venv\Scripts\python.exe validation/curve_spike.py `
  --full ../../models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors `
  --pruned ../../models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors `
  --adapter ../../models/hyperflow/custom_node_hyperflow_8step_v1.0_comfyui.safetensors `
  --output validation/curve_fl2va.json
```

Replace `fl2va` with `ref2va` in the two base paths and output name for the
second family. JSON projection arrays use rows `[video steps 1–8, audio steps
1–8]`, modalities `[video, text, audio]`, then the six groups listed above.
The final layer uses one modality and two groups `[shift, scale]`.

## Phase 2: runtime integration and A/B/C generation gate (2026-09-22)

The revised Option 2 was implemented behind a default-off toggle
(`experimental_curve_refit` on both Apply nodes) and passed the generation
gate. **88 tests pass**; the full-base path is bit-exact versus `main`
(verified across all eight steps, including collision and mask rows).

Integration summary:

- `hyperflow_h3/curve.py`: checkpoint-bound eight-dimensional refits. Fits in
  `assets/curve_fits/*.safetensors` (~34 KiB each) bind `base_sha256`,
  `adapter_sha256`, gate, sigmas, shifts and strength=1.0. Any mismatch
  (unknown checkpoint/adapter, strength != 1, gate/sigma overrides, modified
  MODEL) falls back to backbone-only with a one-line warning. Never assume
  one fit covers every pruned checkpoint.
- `hyperflow_h3/embedder.py`: row-pair expansion factored into `time_pairs`,
  reused by both the full-base path and the curve refit. Bit-exact against
  `main`.
- Pinned `(t,t)` rows: fitted sampled curve (1025 vectors) included;
  final-layer error improves from 4.74% to 1.77% (FL2VA) and 5.45% to 2.40%
  (REF2VA). See `go_no_go` in the JSON reports.
- No ComfyUI core changes; ModelPatcher object/wrapper patches only.

A/B/C generation (`validation/curve_abc.py`): fixed prompt, seed 42, shared
conditioning and initial noise, 768x768, 124 frames, Euler, eight trained
steps. A = full base + HyperFlow (reference), B = pruned backbone-only,
C = pruned + refit. Latent similarity to A (cosine / relative L2):

| Family / stream | B (backbone) | C (refit) |
|---|---|---|
| FL2VA video   | 0.8108 / 0.622 | 0.8818 / 0.483 |
| FL2VA audio   | 0.9800 / 0.200 | 0.9817 / 0.191 |
| REF2VA video  | 0.8656 / 0.515 | 0.8919 / 0.464 |
| REF2VA audio  | 0.9265 / 0.377 | **0.9970 / 0.078** |

C is closer to the full-base reference on every stream of both families
(visually confirmed on FL2VA). Note the absolute gap to A is dominated by
the pruned base itself; the B→C delta is the refit's contribution.

Gate result: **PASS**. The refit ships behind the default-off toggle.

Caveats that remain true: the fit does not reproduce the learned time-LoRA
pathway exactly (residual error relative to the HyperFlow *change* is
~47%/32% in modulation space); single prompt/seed per family; curve path
requires the default recipe (strength 1, default gate/sigmas); new pruned
checkpoints need their own fit generated with `validation/curve_spike.py`
against the exact base and adapter. Decoded clips and per-family metrics:
`validation/abc/`.
