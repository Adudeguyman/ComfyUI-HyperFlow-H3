"""HyperFlow-H3 package: ComfyUI port of Video-Rebirth's 8-step MiniMax-H3 LoRA."""

from hyperflow_h3.weights import (HyperFlowMetadata, HyperFlowWeights, load_weights,
                                  read_metadata, resolve_weights)
from hyperflow_h3.schedule import (DEFAULT_SIGMAS_8STEP, shift_sigmas,
                                   validate_sigmas, video_schedule_sigmas)

__all__ = ["DEFAULT_SIGMAS_8STEP", "HyperFlowMetadata", "HyperFlowWeights",
           "load_weights", "read_metadata", "resolve_weights",
           "shift_sigmas", "validate_sigmas", "video_schedule_sigmas"]
