"""Static and CPU smoke checks for the formal paper experiment matrix."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from custom_modules.register import register_custom_modules
from tools.formal_experiments.protocol import _seal_run
from tools.formal_experiments.registry import ROOT, load_registry, resolve_run
from tools.validate_formal_experiment_notebooks import validate
from tools.windows_collection import verify_checksum_manifest


@pytest.fixture(scope="module", autouse=True)
def _register() -> None:
    register_custom_modules()


def test_registry_has_unique_runs_aliases_outputs_and_required_reuse() -> None:
    registry = load_registry()
    runs = registry["canonical_runs"]
    assert len(runs) == 17
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
    assert resolve_run("M2", registry)[0] == "R13"
    outputs = [run["output_drive_dir"] for run in runs.values()]
    assert len(outputs) == len(set(outputs))


def test_second_dataset_uses_hrsc2016_ms_and_independent_initialization() -> None:
    runs = load_registry()["canonical_runs"]
    assert runs["S00"]["model_yaml"] == runs["R00"]["model_yaml"]
    assert runs["S01"]["model_yaml"].endswith(
        "S01_yolo11n_inceptiondw_dpls_ca_scam_vgup.yaml"
    )
    assert runs["S01"]["module_flags"]["inceptiondw"] is True
    assert runs["S00"]["dataset_id"] == "hrsc2016_ms_yolo_hbb_v1"
    assert runs["S01"]["dataset_id"] == runs["S00"]["dataset_id"]
    assert runs["S00"]["source_archive"].endswith("HRSC2016_MS_YOLO.zip")
    assert runs["S01"]["source_archive"] == runs["S00"]["source_archive"]
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


def test_formal_matrix_scopes_inceptiondw_to_registered_final_adaptations() -> None:
    from ultralytics import YOLO

    runs = load_registry()["canonical_runs"]
    yolov8_expected = runs["R12"]["model_yaml"]
    yolo11_expected = runs["S01"]["model_yaml"]
    paths = {run["model_yaml"] for run in runs.values()}
    for path in paths:
        model = YOLO(str(ROOT / path), verbose=False).model
        types = {
            type(layer).__name__ for layer in model.model
        }
        assert ("C2f_InceptionDW" in types) == (path == yolov8_expected)
        assert ("C3k2_InceptionDW" in types) == (path == yolo11_expected)


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
        "experiments/formal_models/"
        "R12_yolov8n_inceptiondw_dpls_ca_scam_vgup.yaml"
    )
    assert "C2f" in modules
    assert modules.count("C2f_InceptionDW") == 2
    assert "C3k2" not in modules
    assert "C3k2_InceptionDW" not in modules
    assert "C2PSA" not in modules
    assert modules.count("DySample") == 2
    assert modules.count("CASCAM") == 3


def test_yolo11s_baseline_uses_official_s_scale_and_checkpoint() -> None:
    registry = load_registry()["canonical_runs"]
    run = registry["R13"]
    payload = yaml.safe_load(
        (ROOT / run["model_yaml"]).read_text(encoding="utf-8")
    )
    baseline = yaml.safe_load(
        (
            ROOT / "experiments/formal_models/R00_yolo11n_baseline.yaml"
        ).read_text(encoding="utf-8")
    )
    assert run["base_model"] == "YOLO11s"
    assert run["initialization_weight"] == "yolo11s.pt"
    assert payload["scale"] == "s"
    assert payload["backbone"] == baseline["backbone"]
    assert payload["head"] == baseline["head"]


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


def test_formal_notebooks_use_direct_single_seed_chinese_workflow() -> None:
    report = validate()
    assert report["registered_notebooks"] == 17
    assert report["found_notebooks"] == 17
    assert report["passed"]


def test_final_seal_hashes_stable_state_and_removes_running_lock(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "R01" / "seed_0"
    drive_project = tmp_path / "drive_project"
    drive_dir = (
        drive_project / "formal_experiments" / "R01" / "seed_0"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "RUNNING.lock").write_text("running\n", encoding="utf-8")
    (run_dir / "results.csv").write_text(
        "epoch,metric\n0,0.1\n",
        encoding="utf-8",
    )
    drive_dir.mkdir(parents=True)
    (drive_dir / "RUNNING.lock").write_text(
        "stale Drive lock\n",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        run_dir=run_dir,
        drive_dir=drive_dir,
        drive_project_root=str(drive_project),
        run_id="R01",
        run_name="seed_0",
        seed=0,
    )
    manifest = {"run_id": "R01", "seed": 0, "status": "completed"}

    zip_path = _seal_run(config, manifest)

    assert not (run_dir / "RUNNING.lock").exists()
    assert not (drive_dir / "RUNNING.lock").exists()
    state = json.loads(
        (run_dir / "experiment_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "completed"
    checksum_text = (
        run_dir / "artifact_checksums.sha256"
    ).read_text(encoding="utf-8")
    assert "RUNNING.lock" not in checksum_text
    assert "experiment_state.json" in checksum_text
    assert "COMPLETED.ok" in checksum_text
    assert all(
        row["passed"]
        for row in verify_checksum_manifest(
            run_dir / "artifact_checksums.sha256"
        )
    )
    assert all(
        row["passed"]
        for row in verify_checksum_manifest(
            drive_dir / "artifact_checksums.sha256"
        )
    )
    with zipfile.ZipFile(zip_path, "r") as archive:
        assert archive.testzip() is None
