"""Static and simulated-safety validation for all formal Colab notebooks."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import nbformat

from tools.paper_artifacts.formal_protocol import EXPERIMENTS, FormalConfig, restore_or_guard_run

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "colab" / "formal_ablation_v1"
EXPECTED_FILES = {
    "A0_yolo11n": "A0_YOLO11n_Baseline.ipynb",
    "A1_inceptiondw": "A1_InceptionDW.ipynb",
    "A2_inceptiondw_dpls": "A2_InceptionDW_DPLS.ipynb",
    "A3_inceptiondw_dpls_scam": "A3_InceptionDW_DPLS_SCAM.ipynb",
    "A4_inceptiondw_dpls_scam_vgup": "A4_InceptionDW_DPLS_SCAM_VGUP.ipynb",
    "A5_inceptiondw_dpls_ca_scam_vgup": "A5_InceptionDW_DPLS_CA_SCAM_VGUP.ipynb",
}


def _resume_simulation() -> dict[str, bool]:
    with tempfile.TemporaryDirectory(prefix="formal-resume-") as directory:
        root = Path(directory)
        new = FormalConfig(
            experiment_id="A0_yolo11n",
            local_runs_root=str(root / "local"),
            drive_experiment_root=str(root / "drive"),
        )
        checks = {"new": restore_or_guard_run(new) == "new"}
        new.run_dir.mkdir(parents=True)
        (new.run_dir / "weights").mkdir()
        (new.run_dir / "weights" / "last.pt").write_bytes(b"simulation")
        (new.run_dir / "experiment_state.json").write_text(json.dumps({"experiment_id": new.experiment_id}), encoding="utf-8")
        checks["own_resume"] = restore_or_guard_run(new) == "resume"
        (new.run_dir / "experiment_state.json").write_text(json.dumps({"experiment_id": "A1_inceptiondw"}), encoding="utf-8")
        try:
            restore_or_guard_run(new)
        except RuntimeError:
            checks["cross_resume_blocked"] = True
        else:
            checks["cross_resume_blocked"] = False
        (new.run_dir / "COMPLETED.ok").write_text("done", encoding="utf-8")
        try:
            restore_or_guard_run(new)
        except FileExistsError:
            checks["completed_overwrite_blocked"] = True
        else:
            checks["completed_overwrite_blocked"] = False
    return checks


def validate() -> dict[str, Any]:
    reports = []
    commits = set()
    outputs = set()
    forbidden_patterns = [
        (r"gh[pousr]_[A-Za-z0-9_]{20,}", "GitHub token"),
        (r"AIza[0-9A-Za-z_-]{20,}", "Google credential"),
        (r"rm\s+-rf\s+/(?:content/drive|content)?", "broad recursive delete"),
        (r"subprocess\.(?:run|Popen)\([^\n]*train", "subprocess training"),
    ]
    for experiment_id, filename in EXPECTED_FILES.items():
        path = NOTEBOOK_DIR / filename
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        text = "\n".join(cell.source for cell in notebook.cells)
        syntax_errors = []
        for index, cell in enumerate(code_cells):
            try:
                ast.parse(cell.source, filename=f"{filename}:cell-{index}")
            except SyntaxError as error:
                syntax_errors.append(str(error))
        commit_match = re.search(r'FORMAL_CODE_COMMIT\s*=\s*"([0-9a-f]{40})"', text)
        if commit_match:
            commits.add(commit_match.group(1))
        output_match = re.search(r'EXPERIMENT_ID\s*=\s*"([^"]+)"', text)
        if output_match:
            outputs.add(output_match.group(1))
        forbidden = [label for pattern, label in forbidden_patterns if re.search(pattern, text, re.IGNORECASE)]
        training_cells = [cell for cell in code_cells if "formal-training" in cell.metadata.get("tags", [])]
        checks = {
            "experiment_id": f'EXPERIMENT_ID = "{experiment_id}"' in text,
            "fixed_version": '"ultralytics==8.4.92"' in text,
            "central_dataset_config": text.count("DRIVE_DATA_YAML =") == 1,
            "run_training_true": "RUN_TRAINING = True" in text,
            "test_disabled": "RUN_TEST_EVALUATION = False" in text,
            "foreground_training_cell": len(training_cells) == 1 and "train_foreground(" in training_cells[0].source and "subprocess" not in training_cells[0].source,
            "fixed_commit": commit_match is not None,
            "no_forbidden_content": not forbidden,
            "python_syntax": not syntax_errors,
            "drive_output": "ShipPaper/formal_ablation_v1" in text,
            "safe_existing_repo_guard": "start a fresh runtime instead of deleting blindly" in text,
        }
        reports.append({"file": str(path), "checks": checks, "syntax_errors": syntax_errors, "forbidden": forbidden, "passed": all(checks.values())})
    yaml_reports = {}
    for experiment_id, spec in EXPERIMENTS.items():
        path = ROOT / spec["yaml"]
        payload = __import__("yaml").safe_load(path.read_text(encoding="utf-8"))
        yaml_reports[experiment_id] = {
            "path": str(path),
            "exists": path.is_file(),
            "has_sections": all(key in payload for key in ("nc", "backbone", "head")),
        }
    from tools.paper_artifacts.formal_protocol import train_foreground

    training_source = inspect.getsource(train_foreground)
    global_checks = {
        "six_notebooks": len(reports) == 6,
        "one_pinned_commit": len(commits) == 1,
        "unique_experiment_ids": outputs == set(EXPECTED_FILES),
        "unique_output_directories": len(outputs) == 6,
        "all_yamls_valid": all(item["exists"] and item["has_sections"] for item in yaml_reports.values()),
        "official_train_api_direct": "model.train(" in training_source and "subprocess" not in training_source,
    }
    resume = _resume_simulation()
    global_checks["resume_simulation"] = all(resume.values())
    return {
        "notebooks": reports,
        "model_yamls": yaml_reports,
        "resume_simulation": resume,
        "global_checks": global_checks,
        "passed": all(item["passed"] for item in reports) and all(global_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/formal_ablation_v1/notebook_validation.json")
    args = parser.parse_args()
    report = validate()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
