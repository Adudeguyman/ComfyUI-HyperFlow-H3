# Curve fits for pruned MiniMax-H3 bases

Pruned (curve) MiniMax-H3 checkpoints replace the time embedder with a small
`adaln_t_table`, so HyperFlow's two-time `(t, r)` conditioning has nowhere to
go and the pruned build runs backbone-only. A **curve fit** puts most of that
conditioning back: for every AdaLN projection (50 blocks + final layer) it
stores the 8-wide curve coordinates that best reproduce what the full-weight
model produces with HyperFlow, for the 16 trained `(t, r)` pairs and for pinned
`(t, t)` rows. A fit is about 1.7 MB and is bound to one exact pruned base file
and one adapter file. Matching uses the hashes stored inside the fit, never its
file name; fits are named `<pruned base>_hyperflow_curvefit.safetensors`.

Fits for the stock FL2VA/REF2VA pruned bases, the FL2VA/REF2VA hybrids and
Singularity REF2VA pruned v1.3 ship with the node pack in `assets/curve_fits/`
([list](assets/curve_fits/README.md)); for those, just enable the refit.

## Using a fit

1. Bundled fits need nothing. Your own fits go in `models/hyperflow/curve_fits/`
   (any registered `hyperflow` folder works); they take precedence over the
   bundled ones and survive node pack updates.
2. On **Apply HyperFlow H3** / **Apply HyperFlow H3 Advanced**, enable
   `experimental_curve_refit`, keep `strength = 1.0` and leave gate/sigma
   overrides at their defaults.
3. Connect the base loader straight into the Apply node (a ModelSampling node
   is the only thing allowed in between; add other LoRAs after it), and sample
   with Euler on the node's `SIGMAS` output.

The node hashes the base and adapter files (SHA-256) to pick a fit. The first
run of a session takes a few seconds per file; later runs reuse the hash until
the file changes. Matching:

| Base file | Adapter file | Result |
|---|---|---|
| exact hash | exact hash | fit applies, no message |
| exact hash | different | fit applies best-effort, console warning |
| same file name, different hash | any | fit applies best-effort, console warning |
| anything else | any | backbone only: `curve refit disabled: ...` with both hashes |

A fit is only exact for the files it was made from. If you see the best-effort
warning, generate your own fit from your files.

## Making a fit

You need:

- **The pruned base** you load in ComfyUI.
- **The full-weight teacher**: the unpruned base the pruned file was made from,
  in bf16/fp16 or `int8_tensorwise` (the `*_int8_convrot` releases work).
- **A HyperFlow file with the time-embedder LoRAs**: the original
  `minimax_h3_hyperflow_8step_v1.0.safetensors` or the converted full build.
  The `_pruned` build does not contain them.
- **The adapter the Apply node loads** for pruned bases (normally
  `custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors`). Only its hash
  is used.

### With the node

Add **Create HyperFlow H3 Curve Fit (Pruned Bases)** (category `model_patch/video/tools`) on
its own, pick the four files above and queue it. It takes from under a minute to a
few minutes, mostly reading the teacher from disk. It frees about 5 GB of VRAM,
shows progress per projection, can be cancelled, and writes the fit to
`curve_fits/` in the `hyperflow` folder that holds `node_adapter`. The `report`
output (and the console) lists the errors described below; connect it to a
text display node to read it in the UI. For a hybrid, fill `splice` with one
`FILE:FIRST-LAST` line per donor, using names from the diffusion models list.
A ready-made graph is in `example_workflows/hyperflow_curve_fit.json`.

The Apply node looks for fits whenever it runs. ComfyUI caches its output
while its inputs are unchanged, so after making a fit toggle
`experimental_curve_refit` off and on (or restart ComfyUI) to pick it up.

### From the command line

Same fit, without ComfyUI running. From the node pack folder with ComfyUI's
Python (`--device cpu` also works, slower):

```bash
python tools/fit_curve.py \
  --pruned  models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --teacher models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors \
  --hyperflow    models/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors \
  --node-adapter models/hyperflow/custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors
```

The fit is written to
`ComfyUI/models/hyperflow/curve_fits/<pruned base>_hyperflow_curvefit.safetensors`
(override with `--output`).

### Block-spliced hybrids

For a pruned base assembled from several models, describe the matching
full-weight splice (the node's `splice` field takes the same `FILE:FIRST-LAST`
lines). `--teacher` supplies the time embedder, the final layer and
every block not covered by a `--splice FILE:FIRST-LAST`. For example, the
FL2VA base with REF2VA blocks 20-49:

```bash
python tools/fit_curve.py \
  --pruned  models/diffusion_models/minimax_h3_hybrid_fl2va_ref2va_b20-49.safetensors \
  --teacher models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors \
  --splice  models/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors:20-49 \
  --hyperflow    models/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors \
  --node-adapter models/hyperflow/custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors
```

Per-projection fitting matters most here: spliced blocks were pruned against a
different time table than the one the hybrid carries, and a separate fit per
block corrects that without pulling the other blocks off.

### Reading the output

Each projection prints its relative modulation error for the 16 trained steps,
backbone-only versus refit, followed by the mean:

```
blocks.20.adaln_proj.linear: backbone-only 3.43%  refit 1.96%
...
mean relative modulation error: backbone-only 2.06% -> refit 0.88%
```

The refit should be lower on every line. If it is not, the teacher is probably
not the model the pruned file came from. These are modulation errors, not image
quality scores; compare generations with a fixed seed to judge the result.

### Shipping a fit with the node pack

Copy it into `assets/curve_fits/` and add a row to
[its list](assets/curve_fits/README.md). Bind it to the published adapter
(`custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors` from the
HyperFlow Hugging Face repo) so it matches exactly for everyone who downloads
it.

## Limits

- Only the released recipe: 8 steps, default sigmas and gate, strength 1.0.
  Anything else falls back to backbone-only.
- A fit recovers most, not all, of the two-time conditioning. The full base
  with HyperFlow remains the reference.
- Sharing a fit only helps people with byte-identical base and adapter files.

## File format

Safetensors with `generated` `[51, 16, 8]` (video steps 1-8, audio steps 1-8),
`pinned` `[51, 1025, 8]` (pinned `(t, t)` curve on a uniform grid), `t` and `r`
`[16]`, all float32; the first axis is blocks 0-49 then the final layer. The
older shared format without that axis (`[16, 8]` / `[1025, 8]`) applies one
vector set to every projection. Metadata: `format = hyperflow_curve_fit_v1`,
`base_name`, `base_sha256`, `adapter_sha256`, `gate`, `strength`, `sigmas`,
`video_shift`, `audio_shift`, plus informational `teacher` and error fields.
