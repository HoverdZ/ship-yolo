"""Discover and validate the six formal experiment directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.paper_artifacts.formal_protocol import EXPERIMENTS, sha256_file, write_json


def validate_checksums(run_dir: Path) -> dict[str, Any]:
    checksum_file = run_dir / "artifact_checksums.sha256"
    failures = []
    checked = 0
    if checksum_file.is_file():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = run_dir / relative
            checked += 1
            if not path.is_file() or sha256_file(path) != expected:
                failures.append(relative)
    else:
        failures.append("artifact_checksums.sha256")
    return {"checked": checked, "failures": failures, "passed": not failures}


def collect(root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    base = Path(root)
    rows = []
    for experiment_id in EXPERIMENTS:
        run = base / experiment_id
        required = ["run_manifest.json", "best_epoch_summary.json", "complexity.json", "val_image_metrics.csv"]
        missing = [name for name in required if not (run / name).is_file()]
        checksum = validate_checksums(run) if run.is_dir() else {"checked": 0, "failures": ["run directory"], "passed": False}
        rows.append({"experiment_id": experiment_id, "path": str(run), "missing": missing, "checksums": checksum, "complete": not missing and checksum["passed"]})
    report = {"root": str(base), "runs": rows, "all_complete": all(item["complete"] for item in rows)}
    write_json(output or base / "formal_collection_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = collect(args.root, args.output)
    print(json.dumps(report, indent=2))
    if not report["all_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

