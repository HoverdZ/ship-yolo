"""CPU unit checks for ScConv and its C3k2 integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--spatial", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _assert_finite(tensor: Any, label: str) -> None:
    import torch

    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{label} contains NaN or Inf.")


def check_scconv(channels: list[int], spatial: int, device: str) -> list[dict[str, Any]]:
    """Check same-shape forward and finite gradients for representative channels."""

    import torch

    from custom_modules.scconv import ScConv

    results = []
    for channel_count in channels:
        module = ScConv(channel_count).to(device)
        sample = torch.randn(
            1,
            channel_count,
            spatial,
            spatial,
            device=device,
            requires_grad=True,
        )
        output = module(sample)
        if output.shape != sample.shape:
            raise AssertionError(
                f"ScConv({channel_count}) changed shape {tuple(sample.shape)} -> {tuple(output.shape)}."
            )
        _assert_finite(output, f"ScConv({channel_count}) output")
        output.float().square().mean().backward()
        gradients = {
            name: parameter.grad
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        missing = [name for name, gradient in gradients.items() if gradient is None]
        nonfinite = [
            name
            for name, gradient in gradients.items()
            if gradient is not None and not torch.isfinite(gradient).all()
        ]
        if missing or nonfinite:
            raise AssertionError(
                f"ScConv({channel_count}) invalid gradients; missing={missing}, nonfinite={nonfinite}."
            )
        results.append(
            {
                "channels": channel_count,
                "shape": list(output.shape),
                "trainable_tensors": len(gradients),
                "forward": "PASS",
                "backward": "PASS",
                "finite": "PASS",
            }
        )
    return results


def check_c3k2_modes(device: str) -> list[dict[str, Any]]:
    """Compare official and SCConv C3k2 output shapes for both c3k modes."""

    import torch
    from ultralytics.nn.modules.block import Bottleneck, C3k2

    from custom_modules.c3k2_scconv import C3k2_SCConv, SCBottleneck

    results = []
    sample = torch.randn(1, 64, 24, 24, device=device)
    for c3k in (False, True):
        baseline = C3k2(64, 128, n=2, c3k=c3k, e=0.5, shortcut=True).to(device)
        variant = C3k2_SCConv(
            64,
            128,
            n=2,
            c3k=c3k,
            e=0.5,
            shortcut=True,
        ).to(device)
        with torch.no_grad():
            baseline_output = baseline(sample)
            variant_output = variant(sample)
        if baseline_output.shape != variant_output.shape:
            raise AssertionError(
                f"c3k={c3k}: official shape {tuple(baseline_output.shape)} != "
                f"SCConv shape {tuple(variant_output.shape)}."
            )
        _assert_finite(variant_output, f"C3k2_SCConv(c3k={c3k}) output")
        baseline_shortcuts = [
            module.add for module in baseline.modules() if isinstance(module, Bottleneck)
        ]
        variant_shortcuts = [
            module.add
            for module in variant.modules()
            if isinstance(module, SCBottleneck)
        ]
        if baseline.c != variant.c or len(baseline.m) != len(variant.m):
            raise AssertionError(
                f"c3k={c3k}: hidden channels or repeats differ from official C3k2."
            )
        if baseline_shortcuts != variant_shortcuts:
            raise AssertionError(
                f"c3k={c3k}: shortcut flags differ: "
                f"{baseline_shortcuts} != {variant_shortcuts}."
            )
        results.append(
            {
                "c3k": c3k,
                "input_shape": list(sample.shape),
                "output_shape": list(variant_output.shape),
                "hidden_channels": variant.c,
                "repeats": len(variant.m),
                "shortcut_flags": variant_shortcuts,
                "alignment": "PASS",
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if args.spatial <= 0:
        raise ValueError(f"--spatial must be positive, got {args.spatial}.")

    report = {
        "device": args.device,
        "scconv": check_scconv(args.channels, args.spatial, args.device),
        "c3k2_alignment": check_c3k2_modes(args.device),
        "status": "PASS",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
