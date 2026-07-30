"""Benchmark static model complexity under one explicit input resolution."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO, __version__ as ultralytics_version
    from ultralytics.utils.torch_utils import get_flops

    register_custom_modules()
    wrapper = YOLO(args.model, verbose=False)
    network = wrapper.model.eval()
    path = Path(args.model)
    report = {
        "model": args.model,
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "parameters": sum(value.numel() for value in network.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in network.parameters()
            if value.requires_grad
        ),
        "state_tensors": len(network.state_dict()),
        "layers": len(network.model),
        "gflops": float(get_flops(network, imgsz=args.imgsz)),
        "model_size_bytes": path.stat().st_size if path.is_file() else None,
        "detect_strides": [float(value) for value in network.stride],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics_version,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report))
        writer.writeheader()
        writer.writerow(report)
    print(output)


if __name__ == "__main__":
    main()
