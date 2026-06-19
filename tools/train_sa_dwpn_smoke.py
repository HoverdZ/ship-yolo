"""One-epoch smoke training for YOLO11n-SA-DWPN-B.

Use this only after build, forward, and weight-transfer checks pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def default_model_path() -> str:
    repo_yaml = Path("experiments/yolo11n_sa_dwpn_b.yaml")
    source_yaml = Path("ultralytics/cfg/models/11/yolo11n-sa-dwpn.yaml")
    return str(source_yaml if source_yaml.exists() else repo_yaml)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-epoch SA-DWPN smoke train.")
    parser.add_argument("--model", default=default_model_path(), help="Path to SA-DWPN YAML.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Optional pretrained YOLO11n weights.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", default="runs/sa_dwpn")
    parser.add_argument("--name", default="yolo11n_sa_dwpn_smoke")
    parser.add_argument("--exist-ok", action="store_true", default=True)
    args = parser.parse_args()

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
