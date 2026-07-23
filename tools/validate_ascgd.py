"""Validate a trained ASCGD checkpoint on an explicit dataset YAML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ascgd_utils import audit_dataset, register_modules, runtime_versions, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    dataset = audit_dataset(args.data, require_single_class=True)
    register_modules()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(args.data.expanduser().resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        split=args.split,
    )
    payload = {
        "weights": str(weights),
        "dataset": dataset,
        "runtime": runtime_versions(),
        "split": args.split,
        "results": getattr(metrics, "results_dict", None) or {},
        "speed": getattr(metrics, "speed", None) or {},
    }
    output = args.output or weights.parents[1] / f"ascgd_{args.split}_summary.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
