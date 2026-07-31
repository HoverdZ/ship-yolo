"""Validate generated formal Notebooks against registry and safety rules."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.formal_experiments.registry import ROOT, load_registry


def validate(expected_commit: str | None = None) -> dict[str, object]:
    registry = load_registry()
    errors = []
    commits = set()
    found = 0
    for run_id, run in registry["canonical_runs"].items():
        path = ROOT / run["notebook_path"]
        if not path.is_file():
            errors.append(f"{run_id}: Notebook missing: {path}")
            continue
        found += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = payload["cells"]
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in cells
        )
        code_cells = [
            cell for cell in cells if cell.get("cell_type") == "code"
        ]
        described_code_cells = all(
            index > 0
            and cells[index - 1].get("cell_type") == "markdown"
            and re.search(
                r"[\u4e00-\u9fff]",
                "".join(cells[index - 1].get("source", [])),
            )
            for index, cell in enumerate(cells)
            if cell.get("cell_type") == "code"
        )
        training_cells = [
            "".join(cell.get("source", []))
            for cell in code_cells
            if "train_foreground(" in "".join(cell.get("source", []))
        ]
        checks = {
            "run_id": f'RUN_ID = "{run_id}"' in source,
            "model_yaml": run["model_yaml"] in source,
            "three_complete_code_cells": len(code_cells) == 3,
            "chinese_description_before_each_code": described_code_cells,
            "no_manual_training_switch": "RUN_TRAINING" not in source,
            "training_enabled_in_fixed_config": (
                source.count("run_training=True") == 1
            ),
            "single_foreground_training_call": (
                len(training_cells) == 1
                and training_cells[0].count("train_foreground(") == 1
            ),
            "no_manual_seed_variable": "SEED =" not in source,
            "no_multi_seed_loop": not re.search(
                r"for\s+\w*seed\w*\s+in",
                source,
                flags=re.IGNORECASE,
            ),
            "correct_colab_secret": (
                'userdata.get("GITHUB_TOKEN")' in source
                and "SHIP_YOLO_GITHUB_TOKEN" not in source
            ),
            "no_training_subprocess": not re.search(
                r"subprocess\.(?:run|Popen)\([^)]*(?:train|yolo)",
                source,
                flags=re.IGNORECASE | re.DOTALL,
            ),
            "no_old_drive_root": "/MyDrive/ShipPaper" not in source,
            "current_drive_root": (
                "/MyDrive/ship_detection/paper_project" in source
            ),
            "fixed_ultralytics": "ultralytics==8.4.92" in source,
            "safe_resume": "resume=True" in source
            or "train_foreground" in source,
            "direct_after_preflight": (
                "全部训练前检查通过，开始正式训练。" in source
                and "prepared = prepare_experiment(config)" in source
            ),
            "release_training_gpu_before_finalize": (
                'prepared_model = prepared.pop("model", None)' in source
                and (
                    "del prepared_model, trained_model, train_results"
                    in source
                )
                and "torch.cuda.empty_cache()" in source
            ),
            "checksum": "artifact_checksums.sha256" in source,
            "manifest": "run_manifest.json" in source,
            "zip": "exports" in source,
            "no_literal_token": not re.search(
                r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}",
                source,
            ),
        }
        for name, passed in checks.items():
            if not passed:
                errors.append(f"{run_id}: failed {name}")
        matches = re.findall(
            r'FORMAL_CODE_COMMIT = "([0-9a-f]{40})"',
            source,
        )
        if len(matches) != 1:
            errors.append(f"{run_id}: missing/ambiguous fixed commit")
        else:
            commits.add(matches[0])
    if expected_commit and commits != {expected_commit}:
        errors.append(f"Notebook commits {commits} != {expected_commit}")
    protocol = (
        ROOT / "tools" / "formal_experiments" / "protocol.py"
    ).read_text(encoding="utf-8")
    if "results = model.train(" not in protocol:
        errors.append("Protocol does not call official model.train directly.")
    if "subprocess." in protocol.split("def train_foreground", 1)[1].split(
        "def _best_epoch", 1
    )[0]:
        errors.append("train_foreground contains subprocess usage.")
    if "def _seal_run(" not in protocol:
        errors.append("Protocol is missing stable final-state sealing.")
    per_image = (
        ROOT / "tools" / "paper_artifacts" / "per_image_evaluation.py"
    ).read_text(encoding="utf-8")
    if "prediction_batch" not in per_image:
        errors.append("Per-image evaluation is missing bounded batches.")
    if "source=[str(image) for image in images]" in per_image:
        errors.append("Per-image evaluation still submits the full split.")
    report = {
        "registered_notebooks": len(registry["canonical_runs"]),
        "found_notebooks": found,
        "fixed_commits": sorted(commits),
        "errors": errors,
        "passed": not errors,
    }
    if errors:
        raise ValueError("\n".join(errors))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit")
    args = parser.parse_args()
    print(json.dumps(validate(args.commit), indent=2))


if __name__ == "__main__":
    main()
