"""Compare YOLO11n-SCConv-C3k2-Full with the official YOLO11n baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_scconv_modules
from tools.scconv_utils import BASELINE_MODEL, MODEL_YAML, model_statistics, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def _delta(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in baseline:
        difference = variant[key] - baseline[key]
        result[key] = {
            "absolute": difference,
            "percent": difference / baseline[key] * 100 if baseline[key] else 0.0,
        }
    return result


def main() -> None:
    args = parse_args()
    register_scconv_modules()
    from ultralytics import YOLO

    baseline = YOLO(BASELINE_MODEL)
    variant = YOLO(str(MODEL_YAML))
    baseline_stats = model_statistics(baseline, imgsz=args.imgsz)
    variant_stats = model_statistics(variant, imgsz=args.imgsz)
    report = {
        "imgsz": args.imgsz,
        "baseline": {"model": BASELINE_MODEL, **baseline_stats},
        "variant": {
            "model": str(MODEL_YAML.relative_to(ROOT)),
            **variant_stats,
        },
        "delta": _delta(baseline_stats, variant_stats),
        "latency_caveat": (
            "GFLOPs do not predict actual latency. GPU latency, memory, and training "
            "throughput must be measured in Colab."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
