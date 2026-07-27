"""Topology, inheritance, checkpoint, and Colab tests for CA-SCAM."""

from __future__ import annotations

from tools.calibrated_scam_utils import (
    backward_report,
    build_model,
    expected_ca_keys,
    forward_report,
    initialize_from_official,
    state_dict_roundtrip_report,
    structure_report,
)


def test_ca_scam_model_builds_and_runs_cpu() -> None:
    model = build_model(nc=1)
    assert structure_report(model)["passed"]
    assert forward_report(model, imgsz=64)["passed"]
    assert backward_report(imgsz=64)["passed"]


def test_ca_scam_official_initialization_is_audited() -> None:
    model = build_model(nc=1)
    report = initialize_from_official(model, weights="yolo11n.pt", apply=True)
    assert report["passed"], report
    assert report["loaded_total"] == "562/571"
    assert report["missing_new_ca_tensors"] == expected_ca_keys()
    assert report["shared_scam_tensors_loaded"] > 0
    assert all(report["new_ca_zero_initialization"].values())


def test_ca_scam_checkpoint_roundtrip() -> None:
    report = state_dict_roundtrip_report(imgsz=64)
    assert report["passed"], report


def test_ca_notebook_trains_in_current_kernel() -> None:
    from tools.build_ca_scam_notebook import build_notebook

    notebook = build_notebook()
    cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "RUN_TRAINING" in cell.source
    ]
    assert len(cells) == 1
    source = cells[0]
    assert "RUN_TRAINING = True" in source
    assert "train_ca_scam(" in source
    assert 'cache="disk"' in source
    assert "deterministic=False" in source
    assert "subprocess" not in source
    assert "!python" not in source
    assert "%run" not in source
