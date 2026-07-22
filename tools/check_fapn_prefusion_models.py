"""Run all local non-training checks for the two FaPN-Prefusion models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.fapn_prefusion import (
    FaPNAlignmentOnly,
    FaPNDepthwiseModulatedDeformConv2d,
)
from tools.fapn_prefusion_profile import profile_variant
from tools.fapn_prefusion_utils import (
    VARIANTS,
    backward_report,
    build_model,
    compare_parameter_shapes_with_official,
    forward_report,
    semantic_weight_transfer,
    structure_report,
    topology_report,
    validate_init_manifest,
    variant_config,
)


def identity_report() -> dict:
    generator = torch.Generator(device="cpu").manual_seed(23)
    high = torch.randn(2, 128, 12, 12, generator=generator)
    controller = torch.randn(2, 64, 12, 12, generator=generator)
    dcn = FaPNDepthwiseModulatedDeformConv2d(128).eval()
    with torch.inference_mode():
        dcn_output = dcn(high, controller)
    low = torch.randn(2, 128, 12, 12, generator=generator)
    align = FaPNAlignmentOnly(128, 128).eval()
    with torch.inference_mode():
        aligned_output = align([low, high])
    dcn_error = float((dcn_output - high).abs().max())
    align_error = float((aligned_output - high).abs().max())
    checks = {
        "dcn_identity_error_lt_1e_5": dcn_error < 1e-5,
        "alignment_initial_output_error_lt_1e_5": align_error < 1e-5,
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "dcn_max_abs_error": dcn_error,
        "alignment_max_abs_error": align_error,
    }


def check_variant(variant: str, weights: Path, imgsz: int, run_backward: bool) -> dict:
    model = build_model(variant)
    structure = structure_report(model, variant)
    shapes = compare_parameter_shapes_with_official(model)
    transfer = semantic_weight_transfer(model, weights, variant=variant, apply=True)
    forward = forward_report(model, imgsz=imgsz)
    backward = backward_report(model, imgsz=256) if run_backward else {"skipped": True}
    profile = profile_variant(variant, imgsz=imgsz)
    config = variant_config(variant)
    init_validation = {"skipped": True, "reason": "init.pt not present"}
    if Path(config["init_pt"]).is_file() and Path(config["manifest"]).is_file():
        init_validation = validate_init_manifest(config["init_pt"], config["manifest"])
    checks = {
        "structure": structure["all_checks_passed"],
        "unchanged_shapes": shapes["all_checks_passed"],
        "weight_transfer": transfer["all_strict_checks_passed"],
        "forward": forward["all_checks_passed"],
        "backward": backward.get("all_checks_passed", True),
        "profile_positive": profile["parameters"] > 0 and profile["gflops"] > 0,
        "init_manifest": init_validation.get("all_checks_passed", True),
    }
    return {
        "variant": variant,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "structure": structure,
        "unchanged_parameter_shapes": shapes,
        "weight_transfer": transfer,
        "forward": forward,
        "backward": backward,
        "profile": profile,
        "init_manifest": init_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "inceptiondw", "all"], default="all")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    topology = topology_report()
    identity = identity_report()
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    results = {
        "topology": topology,
        "identity": identity,
        "variants": {
            variant: check_variant(
                variant,
                args.weights,
                args.imgsz,
                run_backward=not args.skip_backward,
            )
            for variant in variants
        },
    }
    results["all_checks_passed"] = (
        topology["all_checks_passed"]
        and identity["all_checks_passed"]
        and all(item["all_checks_passed"] for item in results["variants"].values())
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "all_checks_passed": results["all_checks_passed"],
        "topology": topology["all_checks_passed"],
        "identity": identity,
        "variants": {
            variant: {
                "all_checks_passed": payload["all_checks_passed"],
                "checks": payload["checks"],
                "parameters": payload["profile"]["parameters"],
                "gflops": payload["profile"]["gflops"],
                "detect_input_shapes": payload["forward"]["detect_input_shapes"],
                "backward": payload["backward"].get("all_checks_passed", "skipped"),
            }
            for variant, payload in results["variants"].items()
        },
        "detailed_output": str(args.output.resolve()) if args.output else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not results["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
