"""One-epoch smoke training for YOLO11n-SA-DWPN-C-lite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_sa_dwpn_modules


def default_model_path() -> str:
    source_yaml = Path("ultralytics/cfg/models/11/yolo11n-sa-dwpn-c-lite.yaml")
    repo_yaml = Path("experiments/yolo11n_sa_dwpn_c_lite.yaml")
    return str(source_yaml if source_yaml.exists() else repo_yaml)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small SA-DWPN-C-lite smoke train.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--model", default=default_model_path(), help="Path to C-lite YAML.")
    parser.add_argument("--weights", default="", help="Optional initial weights, e.g. SA-DWPN-B best.pt.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/sa_dwpn_c_lite")
    parser.add_argument("--name", default="yolo11n_sa_dwpn_c_lite_smoke")
    parser.add_argument("--resume", default=False)
    parser.add_argument("--exist-ok", action="store_true", default=True)
    args = parser.parse_args()

    register_sa_dwpn_modules()
    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
