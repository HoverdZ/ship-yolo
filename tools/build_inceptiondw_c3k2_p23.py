"""Build and summarize YOLO11n-InceptionDW-C3k2-P23 without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.inceptiondw_utils import MODEL_YAML, build_custom_model, model_statistics, structure_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_YAML), help="Experiment model YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size used for GFLOPs.")
    args = parser.parse_args()

    model = build_custom_model(args.model)
    model.model.info(verbose=True, imgsz=args.imgsz)
    result = {
        "model": str(Path(args.model).resolve()),
        "statistics": model_statistics(model, imgsz=args.imgsz),
        "structure": structure_report(model),
    }
    if not result["structure"]["all_checks_passed"]:
        raise AssertionError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
