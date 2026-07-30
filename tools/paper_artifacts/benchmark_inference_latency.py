"""Measure PyTorch forward latency with a fully recorded benchmark protocol."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _measure(
    network,
    image: torch.Tensor,
    *,
    use_fp16: bool,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    samples = []
    enabled = use_fp16 and image.device.type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(
                device_type=image.device.type,
                enabled=enabled,
                dtype=torch.float16 if enabled else None,
            ):
                network(image)
        if image.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(image.device)
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.autocast(
                device_type=image.device.type,
                enabled=enabled,
                dtype=torch.float16 if enabled else None,
            ):
                network(image)
            if image.is_cuda:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    mean = statistics.fmean(samples)
    return {
        "precision": "FP16 autocast" if enabled else "FP32",
        "warmup": warmup,
        "repeats": repeats,
        "latency_mean_ms": mean,
        "latency_std_ms": statistics.pstdev(samples),
        "latency_p50_ms": float(np.percentile(samples, 50)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "fps": 1000.0 / mean,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(image.device))
            if image.is_cuda
            else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO, __version__ as ultralytics_version

    register_custom_modules()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable.")
    device = torch.device(args.device)
    wrapper = YOLO(args.model, verbose=False)
    network = wrapper.model.eval().to(device)
    image = torch.rand(
        args.batch,
        3,
        args.imgsz,
        args.imgsz,
        device=device,
    )
    rows = [
        _measure(
            network,
            image,
            use_fp16=False,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    ]
    if device.type == "cuda":
        rows.append(
            _measure(
                network,
                image,
                use_fp16=True,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
    common = {
        "model": args.model,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tensorrt": None,
        "ultralytics": ultralytics_version,
        "input_size": args.imgsz,
        "batch": args.batch,
        "includes_preprocessing": False,
        "includes_nms": False,
        "scope": "PyTorch model forward; includes in-model VGUP when present",
    }
    rows = [{**common, **row} for row in rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
