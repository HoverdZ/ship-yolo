"""Probe Torchvision modulated depthwise DCNv2 forward/backward under CUDA AMP."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.fapn_prefusion import FaPNAlignmentOnly


DETERMINISTIC_DCN_WARNING = (
    r".*compute_grad_input does not have a deterministic implementation.*"
)


def run_probe(device: str = "cuda:0", amp: bool = True) -> dict:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA AMP probe requested but torch.cuda.is_available() is false.")
    module = FaPNAlignmentOnly(128, 128).to(requested).train()
    low = torch.randn(1, 128, 32, 32, device=requested, requires_grad=True)
    high = torch.randn(1, 128, 32, 32, device=requested, requires_grad=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=DETERMINISTIC_DCN_WARNING, category=UserWarning)
        with torch.autocast(
            device_type=requested.type,
            dtype=torch.float16 if requested.type == "cuda" else torch.bfloat16,
            enabled=amp and requested.type == "cuda",
        ):
            output = module([low, high])
            loss = output.float().square().mean()
        loss.backward()
    gradients = [parameter.grad for parameter in module.parameters()]
    checks = {
        "output_finite": bool(torch.isfinite(output).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "all_parameter_gradients_present": all(gradient is not None for gradient in gradients),
        "all_parameter_gradients_finite": all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ),
    }
    return {
        "device": str(requested),
        "amp_requested": amp,
        "amp_executed": amp and requested.type == "cuda",
        "torch": torch.__version__,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    result = run_probe(args.device, amp=not args.no_amp)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
