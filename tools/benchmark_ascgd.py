"""Benchmark unfused ASCGD PyTorch inference on a CUDA GPU (intended for L4)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ascgd_utils import (
    REPORT_DIR,
    VARIANTS,
    build_model,
    model_statistics,
    runtime_versions,
    write_json,
)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _measure(
    network: torch.nn.Module,
    x: torch.Tensor,
    *,
    amp: bool,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    network.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp,
            ):
                network(x)
        torch.cuda.synchronize()
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp,
            ):
                network(x)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    mean_ms = statistics.fmean(timings)
    return {
        "precision": "amp_fp16" if amp else "fp32",
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": _percentile(timings, 0.50),
        "p95_latency_ms": _percentile(timings, 0.95),
        "throughput_images_per_second": x.shape[0] * 1000.0 / mean_ms,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "warmup": warmup,
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "l4_benchmark.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_ascgd.py requires a CUDA GPU; run it on Colab L4.")
    if min(args.imgsz, args.batch, args.warmup, args.iterations) <= 0:
        raise ValueError("Benchmark dimensions and iteration counts must be positive.")

    selected = list(VARIANTS) if args.variant == "all" else [args.variant]
    x = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device="cuda")
    results: dict[str, Any] = {}
    for variant in selected:
        yolo = build_model(variant)
        statistics_payload = model_statistics(yolo, imgsz=args.imgsz)
        network = yolo.model.cuda().float().eval()
        results[variant] = {
            "statistics": statistics_payload,
            "fp32": _measure(
                network,
                x,
                amp=False,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "amp_fp16": _measure(
                network,
                x,
                amp=True,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "fused": False,
            "backend": "PyTorch eager",
        }
        print(json.dumps({variant: results[variant]}, indent=2))
        del network, yolo
        torch.cuda.empty_cache()

    payload = {
        "runtime": runtime_versions(),
        "configuration": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "synchronize_each_iteration": True,
            "fused": False,
        },
        "variants": results,
    }
    write_json(args.output, payload)
    csv_path = args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "variant",
            "parameters",
            "gflops",
            "precision",
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "throughput_images_per_second",
            "peak_memory_mib",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, value in results.items():
            for precision in ("fp32", "amp_fp16"):
                measured = value[precision]
                writer.writerow(
                    {
                        "variant": variant,
                        "parameters": value["statistics"]["parameters"],
                        "gflops": value["statistics"]["gflops"],
                        "precision": measured["precision"],
                        "mean_latency_ms": measured["mean_latency_ms"],
                        "p50_latency_ms": measured["p50_latency_ms"],
                        "p95_latency_ms": measured["p95_latency_ms"],
                        "throughput_images_per_second": measured[
                            "throughput_images_per_second"
                        ],
                        "peak_memory_mib": measured["peak_memory_mib"],
                    }
                )
    print(f"Wrote {args.output.resolve()} and {csv_path.resolve()}")


if __name__ == "__main__":
    main()
