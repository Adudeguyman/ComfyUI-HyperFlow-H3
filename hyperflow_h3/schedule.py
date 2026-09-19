"""The HyperFlow sigma grid.

Ported from hyperflow_h3/schedule.py (Apache-2.0; the shift formula derives from
diffusers' MiniMaxH3Scheduler). Only the raw grid and the shift application live
here: the per-row (t, r) planning is ComfyUI-specific and lives in embedder.py.

On ComfyUI the sampler drives the video schedule (ModelSamplingAV carries the
audio stream on the video clock), so the SIGMAS output this module produces is
the raw grid shifted with the video shift -- 9 points, 8 Euler steps.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

#: Raw (unshifted) grid the adapter was trained on: 9 sigma points, 8 model
#: evaluations. Used when the weights file carries no `hyperflow_sigmas` header.
DEFAULT_SIGMAS_8STEP: tuple[float, ...] = (1.0, 0.931506, 0.839236, 0.703462, 0.5,
                                           0.296538, 0.160764, 0.068494, 0.0)

#: Scheduler shifts the adapter was trained with (diffusers applies them per
#: modality; ComfyUI's ModelSamplingAV resolves the audio shift natively).
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def validate_sigmas(sigmas: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Return ``sigmas`` as a float32 CPU tensor after checking it is a valid
    rectified-flow grid: strictly decreasing, starting at or below 1.0, ending at
    exactly 0.0 (the contract diffusers' MiniMaxH3Scheduler.set_timesteps
    enforces for ``sigmas=``)."""
    grid = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
    if grid.numel() < 2:
        raise ValueError(f"A sigma grid needs at least two points, got {grid.numel()}.")
    if not bool((grid[1:] < grid[:-1]).all()):
        raise ValueError(f"The sigma grid must be strictly decreasing, got {grid.tolist()}.")
    if grid[0].item() > 1.0 or grid[-1].item() != 0.0:
        raise ValueError(
            f"The sigma grid must start at or below 1.0 and end at exactly 0.0, got {grid.tolist()}.")
    return grid


def shift_sigmas(sigmas: Sequence[float] | torch.Tensor, shift: float) -> torch.Tensor:
    """The exponential shift ``s * sigma / (1 + (s - 1) * sigma)`` -- the formula
    MiniMaxH3Scheduler uses. Maps 0 to 0 and 1 to 1, so a valid grid stays valid."""
    if shift <= 0:
        raise ValueError(f"`shift` must be positive, got {shift}.")
    base = validate_sigmas(sigmas)
    return shift * base / (1.0 + (shift - 1.0) * base)


def video_schedule_sigmas(raw: Sequence[float] | torch.Tensor | None) -> torch.Tensor:
    """The SIGMAS tensor ComfyUI samplers consume: the raw grid shifted with the
    video shift (shift 12.0), float32 on CPU, ``len == steps + 1`` with 0.0 last."""
    grid = DEFAULT_SIGMAS_8STEP if raw is None else raw
    return shift_sigmas(grid, VIDEO_SHIFT).cpu()


__all__ = ["AUDIO_SHIFT", "DEFAULT_SIGMAS_8STEP", "VIDEO_SHIFT",
           "shift_sigmas", "validate_sigmas", "video_schedule_sigmas"]
