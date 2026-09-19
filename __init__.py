"""ComfyUI node pack for HyperFlow (Video Rebirth's 8-step LoRA for MiniMax-H3).

Reference: github.com/Video-Rebirth/hyperflow (Apache-2.0 loader; the LoRA
weights are a Model Derivative of MiniMax-H3 under the MiniMax H3 Community
License, huggingface.co/videorebirth/hyperflow).

The diffusers weights file is translated onto ComfyUI's native MiniMax-H3
module paths in memory at load time -- no converted copy of the file is
written anywhere. Weights resolve from models/loras/hyperflow/ (registered
below, the same way the VDN-H3 node registers models/vdn).
"""

import os, sys
_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import folder_paths


def _register_folder():
    for base in {os.path.dirname(p) for p in folder_paths.get_folder_paths("loras")}:
        folder_paths.add_model_folder_path("hyperflow", os.path.join(base, "hyperflow"))


if "hyperflow" not in folder_paths.folder_names_and_paths:
    _register_folder()

from hyperflow_h3.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
