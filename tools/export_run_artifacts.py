"""Export a whitelist of lightweight run artifacts for repository archival."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


WHITELIST = {
    "args.yaml",
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "summary.json",
}

EXCLUDED_PARTS = {"weights", "datasets", "data", "cache", "__pycache__"}


def should_copy(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    if path.name in WHITELIST:
        return True
    return path.parent.name in {"val_predictions", "predictions"} and path.suffix.lower() in {".jpg", ".png"} and path.stat().st_size < 2_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight whitelisted YOLO run artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    destination = Path(args.destination)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    destination.mkdir(parents=True, exist_ok=True)

    copied = []
    warnings = []
    for name in sorted(WHITELIST):
        src = run_dir / name
        if not src.exists():
            warnings.append(f"missing: {name}")
            continue
        dst = destination / name
        shutil.copy2(src, dst)
        copied.append(str(dst))

    for path in run_dir.rglob("*"):
        if path.is_file() and should_copy(path) and path.name not in WHITELIST:
            rel = path.relative_to(run_dir)
            dst = destination / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied.append(str(dst))

    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Copied {len(copied)} artifacts to {destination}")


if __name__ == "__main__":
    main()
