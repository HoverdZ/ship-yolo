"""Build and forward-check YOLO11n-SA-DWPN-B.

Run after integrating custom_modules/sa_dwpn.py into an Ultralytics source tree.
This script does not train.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def default_model_path() -> str:
    repo_yaml = Path("experiments/yolo11n_sa_dwpn_b.yaml")
    source_yaml = Path("ultralytics/cfg/models/11/yolo11n-sa-dwpn.yaml")
    return str(source_yaml if source_yaml.exists() else repo_yaml)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check YOLO11n-SA-DWPN-B build and forward.")
    parser.add_argument("--model", default=default_model_path(), help="Path to SA-DWPN YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Dummy input size.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0.")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.info(verbose=True)

    device = torch.device(args.device)
    model.model.to(device)
    model.model.eval()

    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        y = model.model(x)

    sdwf_count = sum(1 for m in model.model.modules() if m.__class__.__name__ == "SDWF")
    detect_count = sum(1 for m in model.model.modules() if m.__class__.__name__ == "Detect")
    print(f"SDWF modules: {sdwf_count}")
    print(f"Detect modules: {detect_count}")
    print(f"Forward OK; output type: {type(y).__name__}")


if __name__ == "__main__":
    main()
