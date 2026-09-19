"""Junction/symlink containment tests for resolve_weights."""
import sys
from pathlib import Path

import pytest
import torch

_COMFYUI_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_COMFYUI_ROOT))
sys.path.insert(0, str(_PACKAGE))

from safetensors.torch import save_file

import hyperflow_h3.weights as weights_mod
from hyperflow_h3.weights import _within_registered, resolve_weights

HEADER = {
    "hyperflow": "true",
    "hyperflow_version": "1.0",
    "hyperflow_gate": "0.5",
    "lora_alpha": "8",
    "base_model": "MiniMaxAI/MiniMax-H3",
    "lora_rank": "4",
}


def _touch_weights(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"blocks.0.mlp.fc2.lora_A.weight": torch.randn(4, 8),
               "blocks.0.mlp.fc2.lora_B.weight": torch.randn(8, 4)},
              str(path), metadata=HEADER)
    return path


def test_within_registered_literal_and_outside(tmp_path):
    root = tmp_path / "hyperflow"
    root.mkdir()
    inside = _touch_weights(root / "sub" / "w.safetensors")
    inside.parent.mkdir(exist_ok=True)
    outside = _touch_weights(tmp_path / "elsewhere.safetensors")
    assert _within_registered(inside, [root])
    assert not _within_registered(outside, [root])
    # a path whose resolution escapes the root still passes on the literal
    # match (that is what keeps junctioned subfolders loadable)
    assert _within_registered(root / "junctioned" / "w.safetensors", [root])


def test_within_registered_resolved_branch(tmp_path):
    """A path NOT literally under the root is accepted only when its resolved
    location is (symlink into the root from outside)."""
    root = tmp_path / "root"
    root.mkdir()
    real = tmp_path / "realstore"
    real.mkdir()
    link = root / "linked.safetensors"
    try:
        link.symlink_to(real / "w.safetensors", target_is_directory=False)
        ok = True
    except (OSError, NotImplementedError):
        ok = False  # no symlink privilege on this box; duck-typed fake below
    if ok:
        _touch_weights(real / "w.safetensors")
        assert _within_registered(link, [root])
    assert not _within_registered(tmp_path / "stray.safetensors", [root])

    class _FakePath:
        """Duck-typed stand-in: literal path + resolved path."""
        def __init__(self, literal, resolved):
            self._literal = Path(literal)
            self._resolved = Path(resolved)

        def is_relative_to(self, other):
            return self._literal.is_relative_to(other)

        def resolve(self, strict=False):
            return self._resolved

    # literal outside the root, resolved inside -> accepted (symlink case)
    assert _within_registered(_FakePath(tmp_path / "x.safetensors",
                                        root / "linked.safetensors"), [root])
    # both outside -> rejected
    assert not _within_registered(_FakePath(tmp_path / "x.safetensors",
                                            tmp_path / "y.safetensors"), [root])


def test_resolve_rejects_escape(tmp_path, monkeypatch):
    """If get_full_path ever hands back a path outside every registered root,
    resolution must raise the ValueError instead of loading it."""
    import folder_paths
    root = tmp_path / "hyperflow"
    root.mkdir()
    folder_paths.add_model_folder_path("hyperflow", str(root))
    evil = _touch_weights(tmp_path / "evil.safetensors")
    monkeypatch.setattr(folder_paths, "get_full_path",
                        lambda folder, name: str(evil))
    with pytest.raises(ValueError, match="leaves its registered model directory"):
        resolve_weights("evil.safetensors")


def test_resolve_accepts_normal_combo(tmp_path):
    import folder_paths
    root = tmp_path / "hyperflow"
    root.mkdir()
    folder_paths.add_model_folder_path("hyperflow", str(root))
    good = _touch_weights(root / "good.safetensors")
    assert resolve_weights("good.safetensors") == good


if __name__ == "__main__":
    import tempfile
    for _ in range(1):
        with tempfile.TemporaryDirectory() as d:
            test_within_registered_literal_and_outside(Path(d))
        with tempfile.TemporaryDirectory() as d:
            test_within_registered_resolved_branch(Path(d), None)
    print("ALL PASS (monkeypatch tests are pytest-only)")
