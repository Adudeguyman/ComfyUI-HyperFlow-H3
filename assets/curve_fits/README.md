# Curve fits

Experimental checkpoint-bound conditioning fits for pruned/curve MiniMax-H3
bases (`experimental_curve_refit` on the Apply nodes, default off).

Each `.safetensors` (~34 KiB) carries 16 fitted `(t, r)` generated rows
(video steps 1–8, audio steps 1–8), a 1025-row fitted pinned curve, and
metadata binding it to an exact pruned base checkpoint + adapter file
(SHA-256), gate 0.25, the trained 8-step sigmas and shifts 12/3, strength 1.0.

Matching is tiered: the exact fitted files apply silently; byte-different
copies of the fitted base or adapter (mirrors, HF downloads) apply best-effort
with a console warning; unknown checkpoints or a modified MODEL fall back to
backbone-only behavior, with the file hashes in the log for support.

- `fl2va.safetensors` — for `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `ref2va.safetensors` — for `minimax_h3_ref2va_pruned_int8_convrot.safetensors`

To fit a new pruned checkpoint, run from the node-pack root with the ComfyUI
venv (see `validation/curve_findings.md` for details and measured error):

```powershell
& C:\Users\drbaph\Documents\ComfyUI\venv\Scripts\python.exe validation/curve_spike.py `
  --full <full base> --pruned <pruned base> `
  --adapter <full adapter> --pruned-adapter <pruned adapter> `
  --output validation/curve_<name>.json --fit assets/curve_fits/<name>.safetensors
```
