"""Static and CPU smoke checks for the formal paper experiment matrix."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from custom_modules.register import register_custom_modules
from tools.formal_experiments.registry import ROOT, load_registry, resolve_run


@pytest.fixture(scope="module", autouse=True)
def _register() -> None:
    register_custom_modules()


def test_registry_has_unique_runs_aliases_outputs_and_required_reuse() -> None:
    registry = load_registry()
    runs = registry["canonical_runs"]
    assert len(runs) == 16
    assert resolve_run("A0", registry)[0] == "R00"
    assert resolve_run("D0", registry)[0] == "R00"
    assert resolve_run("A1", registry)[0] == "R02"
    assert resolve_run("D2", registry)[0] == "R02"
    assert resolve_run("C0", registry)[0] == "R02"
    assert resolve_run("A2", registry)[0] == "R04"
    assert resolve_run("C2", registry)[0] == "R04"
    assert resolve_run("V0", registry)[0] == "R04"
    assert resolve_run("A3", registry)[0] == "R10"
    assert resolve_run("V2", registry)[0] == "R10"
    assert resolve_run("VG3", registry)[0] == "R10"
    outputs = [run["output_drive_dir"] for run in runs.values()]
    assert len(outputs) == len(set(outputs))


def test_second_dataset_reuses_topology_but_not_dataset_or_checkpoint() -> None:
    runs = load_registry()["canonical_runs"]
    assert runs["S00"]["model_yaml"] == runs["R00"]["model_yaml"]
    assert runs["S01"]["model_yaml"] == runs["R10"]["model_yaml"]
    assert runs["S00"]["dataset_id"] != runs["R00"]["dataset_id"]
    assert runs["S01"]["dataset_id"] != runs["R10"]["dataset_id"]
    assert runs["S01"]["initialization_weight"] == "yolo11n.pt"
    assert runs["S01"]["initialization_weight"] != "R10/best.pt"


def test_all_unique_model_yamls_build_forward_and_match_stride() -> None:
    from ultralytics import YOLO

    runs = load_registry()["canonical_runs"]
    checked: set[str] = set()
    for run in runs.values():
        relative = run["model_yaml"]
        if relative in checked:
            continue
        checked.add(relative)
        wrapper = YOLO(str(ROOT / relative), verbose=False)
        wrapper.model.eval()
        with torch.inference_mode():
            output = wrapper.model(torch.zeros(1, 3, 64, 64))
        assert output is not None
        assert [float(value) for value in wrapper.model.stride] == [
            float(value) for value in run["expected_detect_strides"]
        ]
        assert all(
            torch.isfinite(parameter).all()
            for parameter in wrapper.model.parameters()
        )


def test_formal_matrix_excludes_inceptiondw() -> None:
    from ultralytics import YOLO

    paths = {
        run["model_yaml"]
        for run in load_registry()["canonical_runs"].values()
    }
    for path in paths:
        model = YOLO(str(ROOT / path), verbose=False).model
        assert "C3k2_InceptionDW" not in {
            type(layer).__name__ for layer in model.model
        }


def _modules(path: str) -> list[str]:
    payload = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    return [
        str(layer[2])
        for section in ("backbone", "head")
        for layer in payload[section]
    ]


def test_pls_and_dpls_differ_only_in_two_upsamplers() -> None:
    p1 = yaml.safe_load(
        (ROOT / "experiments/formal_models/R01_yolo11n_pls.yaml").read_text(
            encoding="utf-8"
        )
    )
    p2 = yaml.safe_load(
        (ROOT / "experiments/formal_models/R02_yolo11n_dpls.yaml").read_text(
            encoding="utf-8"
        )
    )
    differences = [
        (left, right)
        for left, right in zip(
            p1["backbone"] + p1["head"],
            p2["backbone"] + p2["head"],
            strict=True,
        )
        if left != right
    ]
    assert len(differences) == 2
    assert all(left[2] == "nn.Upsample" for left, _right in differences)
    assert all(right[2] == "DySample" for _left, right in differences)


def test_ca_scam_internal_variants_change_only_calibration_module() -> None:
    paths = (
        "experiments/formal_models/R03_yolo11n_dpls_scam.yaml",
        "experiments/formal_models/R05A_yolo11n_dpls_ca_scam_fixed_beta.yaml",
        "experiments/formal_models/R05B_yolo11n_dpls_ca_scam_unbounded_beta.yaml",
        "experiments/formal_models/R04_yolo11n_dpls_ca_scam.yaml",
    )
    expected = ("SCAM", "CASCAMFixedBeta", "CASCAMUnbounded", "CASCAM")
    for path, module in zip(paths, expected, strict=True):
        names = _modules(path)
        assert names.count(module) == 3
        assert sum(
            names.count(name)
            for name in (
                "SCAM",
                "CASCAMFixedBeta",
                "CASCAMUnbounded",
                "CASCAM",
            )
        ) == 3


def test_vgup_yaml_gate_matrix_is_complete() -> None:
    expected = {
        "R07": [8, 128, False, False],
        "R08": [8, 128, True, False],
        "R09": [8, 128, False, True],
        "R10": [8, 128, True, True],
    }
    runs = load_registry()["canonical_runs"]
    for run_id, args in expected.items():
        payload = yaml.safe_load(
            (ROOT / runs[run_id]["model_yaml"]).read_text(encoding="utf-8")
        )
        first = payload["backbone"][0]
        assert first[2] == "VGUPPreprocessor"
        assert first[3] == args


def test_yolov8_adaptation_keeps_c2f_and_native_channel_scale() -> None:
    modules = _modules(
        "experiments/formal_models/R12_yolov8n_dpls_ca_scam_vgup.yaml"
    )
    assert "C2f" in modules
    assert "C3k2" not in modules
    assert "C2PSA" not in modules
    assert modules.count("DySample") == 2
    assert modules.count("CASCAM") == 3


def test_no_model_yaml_contains_training_parameters() -> None:
    for path in (ROOT / "experiments" / "formal_models").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert not {
            "epochs",
            "batch",
            "optimizer",
            "lr0",
            "data",
        }.intersection(payload)
