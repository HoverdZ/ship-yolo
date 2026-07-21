"""Run structure, CPU-forward, statistics, and weight-transfer checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.inceptiondw_utils import MODEL_YAML, full_check, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_YAML), help="Experiment model YAML.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Official YOLO11n weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="CPU dummy input size.")
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    args = parser.parse_args()

    report = full_check(args.model, args.weights, imgsz=args.imgsz)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit("One or more InceptionDW checks failed.")


if __name__ == "__main__":
    main()
