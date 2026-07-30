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
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        checks = {
            "run_id": f'RUN_ID = "{run_id}"' in source,
            "model_yaml": run["model_yaml"] in source,
            "training_default_off": "RUN_TRAINING = False" in source
            and "RUN_TRAINING = True" not in source,
            "foreground_api": "train_foreground(" in source,
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
