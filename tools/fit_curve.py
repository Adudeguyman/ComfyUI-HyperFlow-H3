"""Command-line front end for hyperflow_h3/curve_fit.py (same as the Create
HyperFlow H3 Curve Fit node). Run from the node pack root with the ComfyUI Python:

    python tools/fit_curve.py --pruned <pruned base> --teacher <full base> \\
        --hyperflow <original or converted full HyperFlow file> \\
        --node-adapter <adapter file the Apply node loads>

Block-spliced hybrids take one --splice per donor, e.g. blocks 20-49 from
the full REF2VA base: --splice minimax_h3_ref2va_int8_convrot.safetensors:20-49
See CURVE_FITS.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_COMFYUI = _PKG.parents[1]
for _p in (str(_COMFYUI), str(_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hyperflow_h3.curve_fit import fit_curve, fit_filename, parse_splice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pruned", required=True, help="pruned/curve base to fit (the file you load in ComfyUI)")
    parser.add_argument("--teacher", required=True, help="full-weight base the pruned file was made from")
    parser.add_argument("--splice", action="append", default=[], type=parse_splice, metavar="FILE:FIRST-LAST",
                        help="take these teacher blocks from another full base (hybrids); repeatable")
    parser.add_argument("--hyperflow", required=True, help="original HyperFlow file or the converted full build")
    parser.add_argument("--node-adapter", required=True, help="adapter file the Apply node loads (usually the _pruned build)")
    parser.add_argument("--output", help="fit file to write (default: models/hyperflow/curve_fits/<pruned name>_hyperflow_curvefit.safetensors)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(fit_curve(args.pruned, args.teacher, args.splice, args.hyperflow, args.node_adapter,
              args.output or _COMFYUI / "models" / "hyperflow" / "curve_fits" / fit_filename(args.pruned), args.device))
