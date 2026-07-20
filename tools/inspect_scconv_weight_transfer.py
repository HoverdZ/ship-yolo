"""Inspect official YOLO11n weight inheritance into the SCConv experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_scconv_modules
from tools.scconv_utils import DEFAULT_WEIGHTS, MODEL_YAML, inspect_weight_transfer, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--model", default=str(MODEL_YAML))
    parser.add_argument("--output", default="", help="Optional full JSON report path.")
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Inspect compatibility without loading matched tensors into the target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_scconv_modules()
    from ultralytics import YOLO

    model = YOLO(args.model)
    report = inspect_weight_transfer(model, args.weights, apply=not args.no_apply)
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(args.output, report)
        print(f"Full report written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
