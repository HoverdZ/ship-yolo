"""Colab training entrypoint for YOLO11n-SCConv-C3k2-Full."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_scconv_modules
from tools.scconv_utils import (
    DEFAULT_WEIGHTS,
    MODEL_YAML,
    inspect_weight_transfer,
    model_statistics,
    write_json,
)


STAGING_FILES = {"resolved_args.json", "weight_transfer.json"}


def parse_bool(value: str | bool) -> bool:
    """Parse common explicit CLI boolean values."""

    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Single-class ship dataset YAML.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/scconv_c3k2_full")
    parser.add_argument("--name", default="yolo11n_scconv_c3k2_full_640")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Resume this exact run; accepts --resume or --resume true.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build, transfer weights, and print reports without starting training.",
    )
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_new_run_directory(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"Run path exists and is not a directory: {run_dir}")
    entries = {entry.name for entry in run_dir.iterdir()}
    if entries and not entries.issubset(STAGING_FILES):
        raise FileExistsError(
            f"Refusing to overwrite existing experiment artifacts in {run_dir}. "
            "Use --resume when weights/last.pt belongs to this exact run."
        )


def _validate_resume_checkpoint(path: Path) -> None:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid resume checkpoint type: {type(checkpoint).__name__}")
    missing = [
        key
        for key in ("epoch", "optimizer", "train_args")
        if key not in checkpoint or checkpoint[key] is None
    ]
    if missing:
        raise RuntimeError(f"Checkpoint {path} is not resumable; missing {missing}.")


def _print_summary(
    args: argparse.Namespace,
    stats: dict[str, Any],
    transfer: dict[str, Any],
) -> None:
    summary = {
        "experiment": "YOLO11n-SCConv-C3k2-Full",
        "model": str(MODEL_YAML),
        "data": str(Path(args.data).resolve()),
        "weights": args.weights,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "model_statistics": stats,
        "weight_transfer": {
            "inherited_state_tensors": transfer["inherited_state_tensors"],
            "target_state_tensors": transfer["target_state_tensors"],
            "parameter_element_ratio": transfer["parameter_element_ratio"],
            "neck_abnormal_unmatched": transfer["neck_abnormal_unmatched"],
            "detect_abnormal_unmatched": transfer["detect_abnormal_unmatched"],
        },
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    for name in ("epochs", "imgsz", "batch", "workers"):
        value = getattr(args, name)
        if name == "workers" and value == 0:
            continue
        _validate_positive(name, value)

    data = Path(args.data)
    if not data.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data}; no coco8 or other fallback will be used."
        )
    run_dir = Path(args.project) / args.name

    register_scconv_modules()
    from ultralytics import YOLO, __version__ as ultralytics_version

    if args.resume:
        last_checkpoint = run_dir / "weights" / "last.pt"
        _validate_resume_checkpoint(last_checkpoint)
        model = YOLO(str(last_checkpoint))
        print(f"Resuming exact run from: {last_checkpoint}")
        model.train(resume=True)
        return

    _validate_new_run_directory(run_dir)
    model = YOLO(str(MODEL_YAML))
    transfer = inspect_weight_transfer(model, args.weights, apply=True)
    stats = model_statistics(model, imgsz=args.imgsz)
    model.info(imgsz=args.imgsz)
    _print_summary(args, stats, transfer)

    if args.dry_run:
        print("Dry run complete; training was not started.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_args = {
        **vars(args),
        "model": str(MODEL_YAML),
        "run_dir": str(run_dir),
        "git_commit": _git_commit(),
        "ultralytics_version": ultralytics_version,
        "model_statistics": stats,
    }
    write_json(run_dir / "resolved_args.json", resolved_args)
    write_json(run_dir / "weight_transfer.json", transfer)

    result = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=True,
    )
    metrics = getattr(result, "results_dict", None) or {}
    print(json.dumps({"status": "completed", "metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
