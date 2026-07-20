#!/usr/bin/env python3
"""Repeat validation of the same checkpoint to detect inference/validation instability."""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def add_repo_root_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def register_project_modules() -> None:
    """Import the repository registration module and call a zero-argument registrar if present."""
    module = importlib.import_module("custom_modules.register")

    for function_name in ("register_custom_modules", "register_modules", "register"):
        function = getattr(module, function_name, None)
        if not callable(function):
            continue

        signature = inspect.signature(function)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if not required:
            function()
            return

    # Some repositories register modules as an import side effect.
    print("Registration module imported; no zero-argument registration function was required.")


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def extract_metrics(metrics: Any) -> dict[str, float]:
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics validation result does not contain metrics.box.")

    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "fitness": float(getattr(metrics, "fitness", box.map)),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = ("precision", "recall", "mAP50", "mAP50_95", "fitness")
    summary: dict[str, dict[str, float]] = {}

    for key in keys:
        values = [row[key] for row in rows]
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "span": max(values) - min(values),
        }

    return summary


def stability_label(max_span: float) -> str:
    if max_span <= 1e-7:
        return "EXACTLY_STABLE"
    if max_span <= 1e-5:
        return "NUMERICALLY_STABLE"
    if max_span <= 1e-3:
        return "SMALL_VARIATION"
    return "SUSPICIOUS_VARIATION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the exact same checkpoint repeatedly and compare metrics."
    )
    parser.add_argument("--weights", required=True, help="Path to best.pt or another checkpoint.")
    parser.add_argument("--data", required=True, help="Dataset YAML used for validation.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Use 0 by default to remove DataLoader worker variability.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--output",
        default="diagnostics/repeat_validation.json",
        help="JSON report path, relative to repository root unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = add_repo_root_to_path()
    register_project_modules()
    set_deterministic(args.seed)

    from ultralytics import YOLO

    weights_path = Path(args.weights).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2.")

    rows: list[dict[str, float]] = []

    for repeat_index in range(1, args.repeats + 1):
        print(f"\n{'=' * 72}")
        print(f"Repeated validation {repeat_index}/{args.repeats}")
        print(f"Checkpoint: {weights_path}")
        print(f"{'=' * 72}")

        # Reload from disk every time so each repeat begins from identical checkpoint state.
        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            augment=False,
            rect=False,
            plots=False,
            save_json=False,
            verbose=False,
            project=str(output_path.parent / "repeat_validation_runs"),
            name=f"repeat_{repeat_index}",
            exist_ok=True,
        )

        row = {"repeat": repeat_index, **extract_metrics(metrics)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        del metrics
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows)
    max_span = max(item["span"] for item in summary.values())
    verdict = stability_label(max_span)

    report = {
        "weights": str(weights_path),
        "data": str(data_path),
        "repeats": args.repeats,
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "seed": args.seed,
            "split": args.split,
        },
        "runs": rows,
        "summary": summary,
        "max_metric_span": max_span,
        "verdict": verdict,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 72}")
    print("Repeated-validation summary")
    print(f"{'=' * 72}")
    for metric_name, values in summary.items():
        print(
            f"{metric_name:10s} "
            f"mean={values['mean']:.9f} "
            f"std={values['std']:.9f} "
            f"min={values['min']:.9f} "
            f"max={values['max']:.9f} "
            f"span={values['span']:.9f}"
        )
    print(f"Verdict: {verdict}")
    print(f"Report: {output_path}")

    if verdict == "SUSPICIOUS_VARIATION":
        print(
            "\nThe same checkpoint produced materially different metrics. "
            "Investigate eval-time randomness, state mutation, custom module behavior, "
            "or non-deterministic validation."
        )
    else:
        print(
            "\nThe same checkpoint is stable. The large epoch-to-epoch jumps therefore "
            "come from checkpoint parameter changes rather than repeated evaluation noise."
        )


if __name__ == "__main__":
    main()
