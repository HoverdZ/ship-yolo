"""Build, topology, checkpoint, and training-entry tests for four models."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.erup_vgup_utils import (
    EXPERIMENTS,
    build_model,
    forward_report,
    state_dict_roundtrip_report,
    structure_report,
)


@pytest.mark.parametrize("experiment", tuple(EXPERIMENTS))
def test_four_adaptive_models_build_and_forward(experiment: str) -> None:
    model = build_model(experiment)
    structure = structure_report(model, experiment)
    forward = forward_report(model, experiment, imgsz=64)
    assert structure["passed"], structure
    assert forward["passed"], forward


@pytest.mark.parametrize("experiment", tuple(EXPERIMENTS))
def test_four_adaptive_model_checkpoints_roundtrip(experiment: str) -> None:
    report = state_dict_roundtrip_report(experiment, imgsz=64)
    assert report["passed"], report


def test_notebook_training_is_direct_official_api_not_subprocess() -> None:
    from tools.build_erup_vgup_notebook import build_notebook

    notebook = build_notebook()
    cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "RUN_TRAINING" in cell.source
    ]
    assert len(cells) == 1
    source = cells[0]
    assert "train_experiment(" in source
    assert "RUN_TRAINING = False" in source
    assert 'cache="disk"' in source
    assert "deterministic=False" in source
    assert "subprocess" not in source
    assert "!python" not in source
    assert "%run" not in source
