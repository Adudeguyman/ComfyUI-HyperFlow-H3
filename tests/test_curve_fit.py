import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from hyperflow_h3.curve import file_hash, read_fit
from hyperflow_h3.curve_fit import fit_curve, parse_splice
from hyperflow_h3.schedule import DEFAULT_SIGMAS_8STEP


def checkpoints(tmp_path):
    torch.manual_seed(7)
    freq, hidden, dim, out = 4, 6, 5, 24
    teacher = {"time_embedder.proj_in.weight": torch.randn(hidden, freq), "time_embedder.proj_in.bias": torch.randn(hidden),
               "time_embedder.proj_out.weight": torch.randn(dim, hidden), "time_embedder.proj_out.bias": torch.randn(dim)}
    donor, pruned = {}, {"adaln_t_table": torch.randn(1025, 8)}
    for name in ("blocks.0.adaln_proj.linear", "blocks.1.adaln_proj.linear", "final_layer.adaln_proj.linear"):
        for weights in (teacher, donor):
            weights[name + ".weight"], weights[name + ".bias"] = torch.randn(out, dim), torch.randn(out)
        pruned[name + ".weight"], pruned[name + ".bias"] = torch.randn(out, 8), torch.randn(out)
    hyperflow = {}
    for branch in ("time_embedder", "endpoint_time_embedder"):
        for layer, (i, o) in (("linear_1", (freq, hidden)), ("linear_2", (hidden, dim))):
            hyperflow[f"transformer.{branch}.{layer}.lora_A.weight"] = torch.randn(2, i) * .1
            hyperflow[f"transformer.{branch}.{layer}.lora_B.weight"] = torch.randn(o, 2) * .1
    meta = {"hyperflow": "true", "lora_alpha": "2", "hyperflow_gate": "0.25",
            "hyperflow_sigmas": json.dumps(DEFAULT_SIGMAS_8STEP)}
    paths = {}
    for name, tensors, metadata in (("teacher", teacher, None), ("donor", donor, None), ("pruned", pruned, None),
                                    ("hyperflow", hyperflow, meta)):
        paths[name] = tmp_path / f"{name}.safetensors"
        save_file(tensors, paths[name], metadata=metadata)
    paths["adapter"] = tmp_path / "adapter.safetensors"
    paths["adapter"].write_bytes(b"node adapter")
    return paths


def test_fit_writes_a_bound_per_projection_fit(tmp_path):
    paths = checkpoints(tmp_path)
    output = tmp_path / "fits" / "pruned.safetensors"
    calls = []
    report = fit_curve(paths["pruned"], paths["teacher"], [], paths["hyperflow"], paths["adapter"],
                       output, "cpu", lambda: calls.append(1))
    data, meta = read_fit(output)
    assert data["generated"].shape == (3, 16, 8) and data["pinned"].shape == (3, 1025, 8)
    assert meta["base_sha256"] == file_hash(paths["pruned"]) and meta["adapter_sha256"] == file_hash(paths["adapter"])
    assert float(meta["refit_error"]) < float(meta["backbone_error"])
    assert len(calls) == 3 and "wrote" in report

    spliced = tmp_path / "spliced.safetensors"
    fit_curve(paths["pruned"], paths["teacher"], [(paths["donor"], 1, 1)], paths["hyperflow"], paths["adapter"],
              spliced, "cpu")
    other, _ = read_fit(spliced)
    torch.testing.assert_close(other["generated"][0], data["generated"][0])
    assert not torch.allclose(other["generated"][1], data["generated"][1])


def test_fit_reads_prefixed_checkpoints(tmp_path):
    paths = checkpoints(tmp_path)
    fit_curve(paths["pruned"], paths["teacher"], [], paths["hyperflow"], paths["adapter"], tmp_path / "a.safetensors", "cpu")
    for name in ("teacher", "pruned"):
        tensors = load_file(paths[name])
        save_file({"model.diffusion_model." + k: v for k, v in tensors.items()}, paths[name])
    fit_curve(paths["pruned"], paths["teacher"], [], paths["hyperflow"], paths["adapter"], tmp_path / "b.safetensors", "cpu")
    torch.testing.assert_close(read_fit(tmp_path / "b.safetensors")[0]["generated"],
                               read_fit(tmp_path / "a.safetensors")[0]["generated"])


def test_fit_rejects_wrong_files(tmp_path):
    paths = checkpoints(tmp_path)
    with pytest.raises(ValueError, match="not a pruned"):
        fit_curve(paths["teacher"], paths["teacher"], [], paths["hyperflow"], paths["adapter"], tmp_path / "x", "cpu")
    with pytest.raises(ValueError, match="not a full-weight"):
        fit_curve(paths["pruned"], paths["pruned"], [], paths["hyperflow"], paths["adapter"], tmp_path / "x", "cpu")


def test_parse_splice():
    assert parse_splice("sub/ref2va.safetensors:20-49") == ("sub/ref2va.safetensors", 20, 49)
    assert parse_splice(" ref2va.safetensors:7 ") == ("ref2va.safetensors", 7, 7)
    with pytest.raises(ValueError, match="FILE:FIRST-LAST"):
        parse_splice("ref2va.safetensors")
