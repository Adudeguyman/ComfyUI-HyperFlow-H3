"""Schedule + weights loading tests (CPU)."""
import sys
from pathlib import Path

import pytest
import torch

_COMFYUI_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_COMFYUI_ROOT))
sys.path.insert(0, str(_PACKAGE))

from safetensors.torch import save_file

from hyperflow_h3.schedule import (DEFAULT_SIGMAS_8STEP, shift_sigmas,
                                   validate_sigmas, video_schedule_sigmas)
from hyperflow_h3.weights import (load_weights, read_metadata, resolve_weights)

HEADER = {
    "hyperflow": "true",
    "hyperflow_version": "1.0",
    "hyperflow_gate": "0.5",
    "lora_alpha": "8",
    "base_model": "MiniMaxAI/MiniMax-H3",
    "lora_rank": "4",
    "hyperflow_sigmas": "[1.0, 0.5, 0.0]",
}


def _write_converted(path, extra_keys=None, header=None):
    keys = {
        "blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(12, 8),
        "blocks.0.attn.qkv_proj.lora_B.weight": torch.randn(30, 12),
        "blocks.0.mlp.fc2.lora_A.weight": torch.randn(4, 8),
        "blocks.0.mlp.fc2.lora_B.weight": torch.randn(8, 4),
        "time_embedder.proj_in.lora_A.weight": torch.randn(4, 6),
        "time_embedder.proj_in.lora_B.weight": torch.randn(8, 4),
        "endpoint_time_embedder.proj_in.lora_A.weight": torch.randn(4, 6),
        "endpoint_time_embedder.proj_in.lora_B.weight": torch.randn(8, 4),
    }
    keys.update(extra_keys or {})
    save_file(keys, str(path), metadata=header or HEADER)
    return path


# --- schedule -----------------------------------------------------------------

def test_validate_sigmas():
    grid = validate_sigmas([1.0, 0.5, 0.0])
    assert grid.dtype == torch.float32 and grid.is_cpu
    for bad in ([0.5], [1.0, 0.5, 0.6, 0.0], [1.1, 0.0], [1.0, 0.5]):
        with pytest.raises(ValueError):
            validate_sigmas(bad)


def test_shift_sigmas_known_values():
    got = shift_sigmas([1.0, 0.5, 0.0], 12.0)
    assert got[0].item() == pytest.approx(1.0)
    assert got[1].item() == pytest.approx(12 * 0.5 / (1 + 11 * 0.5))
    assert got[2].item() == 0.0
    assert shift_sigmas([1.0, 0.0], 12.0)[1].item() == 0.0


def test_video_schedule_sigmas():
    sigmas = video_schedule_sigmas(None)
    assert sigmas.numel() == len(DEFAULT_SIGMAS_8STEP)
    assert sigmas[-1].item() == 0.0
    assert bool((sigmas[1:] < sigmas[:-1]).all())
    assert sigmas[0].item() == pytest.approx(1.0)
    # the file grid wins when present
    custom = video_schedule_sigmas([1.0, 0.25, 0.0])
    assert custom.numel() == 3 and custom[-1].item() == 0.0


# --- weights ------------------------------------------------------------------

def test_load_converted(tmp_path):
    path = _write_converted(tmp_path / "hyperflow_test.safetensors")
    w = load_weights(path)
    assert set(w.lora) == {"blocks.0.attn.qkv_proj", "blocks.0.mlp.fc2",
                           "time_embedder.proj_in"}
    assert set(w.endpoint) == {"proj_in"}
    assert w.rank == 4
    # fused qkv entries carry 3x alpha so the per-part scale stays alpha/rank
    assert w.alphas["blocks.0.attn.qkv_proj"] == pytest.approx(3 * 8)
    assert w.alphas["blocks.0.mlp.fc2"] == pytest.approx(8)
    assert w.metadata.gate == 0.5
    assert w.metadata.sigmas == (1.0, 0.5, 0.0)


def test_load_pruned_variant(tmp_path):
    keys = {
        "blocks.0.mlp.fc2.lora_A.weight": torch.randn(4, 8),
        "blocks.0.mlp.fc2.lora_B.weight": torch.randn(8, 4),
    }
    path = tmp_path / "hyperflow_pruned_test.safetensors"
    save_file(keys, str(path), metadata=HEADER)
    w = load_weights(path)
    assert w.endpoint == {}
    assert set(w.lora) == {"blocks.0.mlp.fc2"}


def test_reject_original_layout(tmp_path):
    keys = {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(4, 8),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(8, 4),
    }
    path = tmp_path / "original.safetensors"
    save_file(keys, str(path), metadata=HEADER)
    with pytest.raises(RuntimeError, match="ORIGINAL diffusers-layout"):
        load_weights(path)


def test_metadata_validation(tmp_path):
    path = _write_converted(tmp_path / "bad.safetensors",
                            header={"hyperflow": "true", "hyperflow_version": "1.0"})
    with pytest.raises(ValueError):
        read_metadata(path)


def test_bad_pair_shapes(tmp_path):
    path = _write_converted(
        tmp_path / "shapes.safetensors",
        extra_keys={"blocks.1.mlp.fc1.lora_A.weight": torch.randn(4, 9),
                    "blocks.1.mlp.fc1.lora_B.weight": torch.randn(8, 5)})
    with pytest.raises(ValueError):
        load_weights(path)


def test_resolve_single_file_dir(tmp_path):
    path = _write_converted(tmp_path / "only.safetensors")
    assert resolve_weights(tmp_path) == path
    assert resolve_weights(str(path)) == path


def test_resolve_combo_name(tmp_path):
    import folder_paths
    folder_paths.add_model_folder_path("hyperflow", str(tmp_path))
    path = _write_converted(tmp_path / "combo.safetensors")
    assert resolve_weights("combo.safetensors") == path
    with pytest.raises(FileNotFoundError):
        resolve_weights("nope.safetensors")


def test_download_on_demand(tmp_path, monkeypatch):
    """download_if_missing fetches exactly the manifest-named file via the HF
    cache; repo id + filename come from fixed allowlists, never free text.
    The folder lookup is patched to the tmp dir -- this test must never write
    into a real model folder."""
    import folder_paths
    import hyperflow_h3.weights as weights_mod
    monkeypatch.setattr(folder_paths, "get_folder_paths",
                        lambda name: [str(tmp_path)])
    calls = []

    def fake_download(repo_id, filename, dest_dir):
        assert Path(dest_dir).resolve() == tmp_path.resolve()
        calls.append((repo_id, filename))
        out = dest_dir / filename
        save_file({"blocks.0.mlp.fc2.lora_A.weight": torch.randn(4, 8),
                   "blocks.0.mlp.fc2.lora_B.weight": torch.randn(8, 4)},
                  str(out), metadata=HEADER)
        return out

    monkeypatch.setattr(weights_mod, "_download", fake_download)
    got = weights_mod.ensure_downloaded("pruned", repo_id="drbaph/Hyperflow-Comfyui")
    assert calls == [("drbaph/Hyperflow-Comfyui",
                      "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors")]
    assert got.name == "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors"
    assert got.is_file()
    # the downloaded file loads straight away
    w = load_weights(got)
    assert set(w.lora) == {"blocks.0.mlp.fc2"} and w.endpoint == {}
    with pytest.raises(ValueError):
        weights_mod.manifest_filename("bogus')")


if __name__ == "__main__":
    import tempfile
    test_validate_sigmas()
    test_shift_sigmas_known_values()
    test_video_schedule_sigmas()
    with tempfile.TemporaryDirectory() as d:
        test_load_converted(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_load_pruned_variant(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_reject_original_layout(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_metadata_validation(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_bad_pair_shapes(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_resolve_single_file_dir(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_resolve_combo_name(Path(d))
    print("ALL PASS")
