"""Pytest path setup for the HyperFlow-H3 suite.

The ComfyUI checkout root must be importable (hyperflow_h3 imports comfy
modules), and the node pack root must be importable as `hyperflow_h3`.
Works for both `pytest tests/` and standalone `python tests/test_x.py`.
"""
import sys
from pathlib import Path

_COMFYUI_ROOT = Path(__file__).resolve().parents[3]  # the ComfyUI checkout
_PACKAGE = Path(__file__).resolve().parents[1]       # this node pack
for _p in (str(_COMFYUI_ROOT), str(_PACKAGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
