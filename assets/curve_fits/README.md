# Bundled curve fits

Per-block HyperFlow curve fits for pruned MiniMax-H3 bases, used when
`experimental_curve_refit` is on. Each file is bound by SHA-256 to one pruned
base and to the published adapter
`custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors` (`0b274d1c…`).
Matching uses those hashes, not these file names.

| Fit | Pruned base | Teacher | Error (backbone-only → refit) |
|---|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot_hyperflow_curvefit` | `minimax_h3_fl2va_pruned_int8_convrot` | `minimax_h3_fl2va_int8_convrot` | 1.37% → 0.58% |
| `minimax_h3_ref2va_pruned_int8_convrot_hyperflow_curvefit` | `minimax_h3_ref2va_pruned_int8_convrot` | `minimax_h3_ref2va_int8_convrot` | 2.63% → 0.58% |
| `minimax_h3_hybrid_fl2va_ref2va_b15-49_hyperflow_curvefit` | `minimax_h3_hybrid_fl2va_ref2va_b15-49` | FL2VA + REF2VA blocks 15-49 | 2.23% → 1.00% |
| `minimax_h3_hybrid_fl2va_ref2va_b20-49_hyperflow_curvefit` | `minimax_h3_hybrid_fl2va_ref2va_b20-49` | FL2VA + REF2VA blocks 20-49 | 2.06% → 0.88% |
| `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8_hyperflow_curvefit` | `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8` | FL2VA + REF2VA blocks 25-49 | 1.90% → 0.77% |
| `minimax_h3_hybrid_fl2va_ref2va_b30-49-int8_hyperflow_curvefit` | `minimax_h3_hybrid_fl2va_ref2va_b30-49-int8` | FL2VA + REF2VA blocks 30-49 | 1.76% → 0.68% |
| `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8_hyperflow_curvefit` | `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8` | `Minimax-h3_Singularity_ref2va_v1.3_int8` | 1.87% → 0.76% |

Errors are mean relative modulation errors over the 8 trained video and audio
steps, not image-quality scores. To make a fit for another base, or to add one
here, see [`CURVE_FITS.md`](../../CURVE_FITS.md).
