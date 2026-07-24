"""Validate all CrossConv/DD/CGFM models on CPU without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.module_ablation_utils import CONTROL_EXPERIMENT, EXPERIMENTS, validate_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/module_ablation_validation.json",
    )
    parser.add_argument("--include-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Official checkpoint not found: {args.weights}")
    selected = dict(EXPERIMENTS)
    if args.include_control:
        selected.update(CONTROL_EXPERIMENT)
    reports = {}
    for name, model_yaml in selected.items():
        print(f"[validate] {name}: {model_yaml}")
        reports[name] = validate_experiment(name, model_yaml, args.weights, args.imgsz)
        summary = reports[name]
        stats = summary["statistics"]
        transfer = summary["transfer"]
        print(
            f"  params={stats['parameters']:,}, GFLOPs={stats['gflops']:.3f}, "
            f"transfer={transfer['matched_tensors']}/{transfer['target_tensors']} "
            f"({transfer['tensor_match_ratio']:.2%}), CPU={summary['cpu_forward']['passed']}"
        )

    payload = {
        "imgsz": args.imgsz,
        "weights": (
            args.weights.resolve().relative_to(ROOT).as_posix()
            if args.weights.resolve().is_relative_to(ROOT)
            else str(args.weights)
        ),
        "gpu_benchmark": "not tested",
        "formal_training_started": False,
        "models": reports,
        "all_passed": all(report["passed"] for report in reports.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {args.output}")
    if not payload["all_passed"]:
        raise SystemExit("One or more module ablation checks failed.")


if __name__ == "__main__":
    main()
