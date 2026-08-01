"""Static validation for both InceptionDW CA-SCAM Colab Notebooks."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.formal_experiments.registry import ROOT
from tools.incdw_ca_scam_experiments import (
    RUN_IDS,
    load_incdw_ca_scam_registry,
)


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def validate_notebook(path: Path, run_id: str) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    types = [cell["cell_type"] for cell in cells]
    assert types == [
        "markdown",
        "markdown",
        "code",
        "markdown",
        "code",
        "markdown",
        "code",
    ], (path, types)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)

    setup, training, finalization = map(_source, code_cells)
    all_source = "\n".join(map(_source, cells))
    commit_match = re.search(
        r'FORMAL_CODE_COMMIT = "([0-9a-f]{40})"',
        setup,
    )
    assert commit_match, f"{path}: missing fixed commit"
    assert f'RUN_ID = "{run_id}"' in setup
    assert 'userdata.get("GITHUB_TOKEN")' in setup
    assert "%pip install -q ultralytics==8.4.92" in setup
    assert 'ultralytics.__version__ == "8.4.92"' in setup
    assert '"fetch"' in setup and '"--depth=1"' in setup
    assert "git pull" not in all_source
    assert "GITHUB_TOKEN=" not in all_source

    assert (
        "build_incdw_ca_scam_config(RUN_ID, run_training=True)"
        in training
    )
    assert "prepare_incdw_ca_scam_experiment(config)" in training
    assert "train_foreground(" in training
    assert "subprocess" not in training
    assert "inference_mode" not in training
    assert "RUN_TRAINING" not in all_source
    assert "config.copy_workers == 32" in training
    assert "FROZEN_TRAINING.items()" in training
    assert '"yolo11n.pt"' in training
    assert "torch.cuda.empty_cache()" in training

    assert "finalize_run(" in finalization
    assert 'globals().get("drive_mirror")' in finalization
    assert "update_incdw_ca_scam_comparison" in finalization
    assert "verify_checksum_manifest" in finalization
    assert "RUNNING.lock" in finalization

    for source in (setup, training, finalization):
        python_source = "\n".join(
            "pass" if line.lstrip().startswith("%") else line
            for line in source.splitlines()
        )
        ast.parse(python_source)
    return {
        "run_id": run_id,
        "path": str(path),
        "cells": len(cells),
        "code_cells": len(code_cells),
        "fixed_commit": commit_match.group(1),
        "foreground_training": True,
        "passed": True,
    }


def validate_all(*, root: Path = ROOT) -> dict:
    registry = load_incdw_ca_scam_registry()
    reports = []
    for run_id in RUN_IDS:
        path = root / registry["experiments"][run_id]["notebook_path"]
        reports.append(validate_notebook(path, run_id))
    commits = {report["fixed_commit"] for report in reports}
    assert len(commits) == 1, commits
    return {
        "notebooks": reports,
        "fixed_commit": next(iter(commits)),
        "passed": len(reports) == len(RUN_IDS)
        and all(report["passed"] for report in reports),
    }


def main() -> None:
    print(json.dumps(validate_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
