"""Comfy compiler (aimdo malloc-graph) workaround, ported from the VDN-H3 node.

Comfy builds from 2026-09-04 ship a model compiler + aimdo malloc-graph that
hard-fails on patched MiniMax-H3 forwards (graph breaks raise 'aimdo memory
compile error'; some paths abort the process mid-step). This node does not need
that compiler, so on affected builds an APPLY_MODEL wrapper switches it off --
the same effect as launching with --disable-comfy-compiler, scoped to this
model's calls only (restored in a finally after every call), so unpatched
workflows keep it. No-op on builds without the compiler stack.
"""

from __future__ import annotations

import logging

import comfy.cli_args
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.patcher_extension import WrappersMP

_log = logging.getLogger("comfy.hyperflow")

_WARNED = False


def needs_compiler_workaround() -> bool:
    """Detect only: a failed node application must not change a global flag.
    Returns True when the APPLY_MODEL wrapper needs to manage the switch."""
    global _WARNED
    try:
        args = comfy.cli_args.args
        if not hasattr(args, "disable_comfy_compiler"):
            return False
        aimdo = getattr(comfy.model_prefetch, "comfy_aimdo", None)
        if aimdo is None or not hasattr(aimdo, "malloc_graph"):
            return False
        if getattr(args, "disable_comfy_compiler", False):
            return False
        if not _WARNED:
            _WARNED = True
            _log.warning(
                "[hyperflow] this comfy build's model compiler crashes with patched "
                "MiniMax-H3 (aimdo malloc-graph); disabling it while HyperFlow is "
                "sampling. Remove this once comfy fixes the compiler.")
        return True
    except Exception:
        return False


def _without_comfy_compiler(executor, *args, **kwargs):
    # MiniMax starts its allocation graph before DIFFUSION_MODEL wrappers.
    # APPLY_MODEL encloses that outer forward too, including graph cleanup.
    previous = comfy.cli_args.args.disable_comfy_compiler
    comfy.cli_args.args.disable_comfy_compiler = True
    try:
        return executor(*args, **kwargs)
    finally:
        comfy.cli_args.args.disable_comfy_compiler = previous


def install_compiler_workaround(new_model):
    if needs_compiler_workaround():
        new_model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "hyperflow_compiler",
                                       _without_comfy_compiler)


__all__ = ["install_compiler_workaround", "needs_compiler_workaround"]
