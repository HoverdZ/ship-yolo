"""Model complexity and repeatable inference-latency profiling."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from tools.paper_artifacts.formal_protocol import FormalConfig, write_json


def _latency(network, image: torch.Tensor, *, amp: bool, warmup: int = 20, repeats: int = 100) -> dict[str, Any]:
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type=image.device.type, enabled=amp):
                network(image)
        if image.is_cuda:
            torch.cuda.synchronize()
        for _ in range(repeats):
            start = time.perf_counter()
            with torch.autocast(device_type=image.device.type, enabled=amp):
                network(image)
            if image.is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times),
        "warmup": warmup,
        "repeats": repeats,
        "scope": "PyTorch model forward only; excludes image decode, preprocessing, NMS, and transfer",
    }


def profile_model(config: FormalConfig, model) -> dict[str, Any]:
    from ultralytics.utils.torch_utils import get_flops

    network = model.model.eval()
    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() and str(config.device) != "cpu" else "cpu")
    network.to(device)
    image = torch.rand(1, 3, config.imgsz, config.imgsz, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fp32 = _latency(network, image, amp=False)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    amp = _latency(network, image, amp=True) if device.type == "cuda" else None
    best = config.run_dir / "weights" / "best.pt"
    report = {
        "parameters": sum(value.numel() for value in network.parameters()),
        "trainable_parameters": sum(value.numel() for value in network.parameters() if value.requires_grad),
        "state_tensors": len(network.state_dict()),
        "layers": len(network.model),
        "gflops": float(get_flops(network, imgsz=config.imgsz)),
        "model_size_bytes": best.stat().st_size if best.is_file() else None,
        "detect_strides": [float(value) for value in network.stride],
        "input_shape": list(image.shape),
        "peak_gpu_memory_bytes": peak,
        "pytorch_fp32": fp32,
        "pytorch_amp_fp16": amp,
        "benchmark_warning": "Colab latency is a controlled research measurement, not an industrial deployment claim.",
    }
    write_json(config.run_dir / "complexity.json", report)
    (config.run_dir / "model_summary.txt").write_text(str(network) + "\n\n" + json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["profile_model"]

