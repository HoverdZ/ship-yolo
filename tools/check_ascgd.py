"""Run the complete ASCGD CPU/CUDA preflight and write auditable reports."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ascgd_utils import (
    DEFAULT_WEIGHTS,
    REPORT_DIR,
    VARIANTS,
    backbone_fairness,
    baseline_equivalence,
    build_model,
    cuda_amp_check,
    forward_signature,
    gradient_check,
    model_statistics,
    module_inventory,
    runtime_versions,
    transfer_weights,
    window_padding_check,
    write_json,
)


FORBIDDEN_TOKENS = (
    "fsm",
    "fam",
    "deform",
    "dysample",
    "skconv",
    "bifpn",
    "tood",
    "flash",
    "xformers",
)


def _write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "yaml",
        "parameters",
        "trainable_parameters",
        "gflops",
        "gflops_error",
        "delta_parameters_vs_a",
        "delta_gflops_vs_a",
        "delta_parameters_vs_e",
        "delta_gflops_vs_e",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ASCGD preflight report",
        "",
        f"Overall status: **{'PASS' if summary['all_checks_passed'] else 'FAIL'}**",
        "",
        "Formal 150-epoch training was not started.",
        "",
        "## Variant checks",
        "",
        "| Variant | Build/forward | Detect shapes | Finite | Params | GFLOPs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, result in summary["variants"].items():
        checks = result["checks"]
        stats = result["statistics"]
        gflops = (
            f"{stats['gflops']:.6f}"
            if stats["gflops"] is not None
            else f"ERROR: {stats['gflops_error']}"
        )
        lines.append(
            f"| {variant} | {checks['forward']} | {checks['detect_shapes']} | "
            f"{checks['finite']} | {stats['parameters']} | {gflops} |"
        )
    lines.extend(
        [
            "",
            "## Cross-variant checks",
            "",
            f"- A equals validated InceptionDW baseline: "
            f"{summary['baseline_equivalence']['all_checks_passed']}",
            f"- All backbones match: {summary['backbone_fairness']['all_checks_passed']}",
            f"- Window padding/reverse and non-standard shape check: "
            f"{summary['window_check_passed']}",
            f"- Rectangular E forward (non-window-multiple feature sizes): "
            f"{summary['rectangular_forward_passed']}",
            f"- E backward gradients present and finite: "
            f"{summary['gradient_check_passed']}",
            f"- E inherited parameter elements: "
            f"{summary['weight_transfer_summary']['e_full']['inherited_parameter_elements']}/"
            f"{summary['weight_transfer_summary']['e_full']['target_parameter_elements']} "
            f"(backbone "
            f"{summary['weight_transfer_summary']['e_full']['backbone_parameter_inheritance_ratio']:.4%}, "
            f"Detect "
            f"{summary['weight_transfer_summary']['e_full']['detect_parameter_inheritance_ratio']:.4%})",
            f"- CUDA AMP: {summary['cuda_amp']['status']}",
            "",
            "## Profiling note",
            "",
            "GFLOPs use Ultralytics 8.4.92 THOP conventions. Standard convolution "
            "operators are counted; THOP may not include the explicit attention "
            "matrix multiplications. Any profiler exception is recorded rather "
            "than silently skipped.",
            "",
            "## Remaining GPU-only validation",
            "",
            "L4 AMP stability, FP32/FP16 latency, throughput, peak memory, and "
            "formal accuracy remain unmeasured on this CPU-only host.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    selected = list(VARIANTS) if args.all or not args.variant else [args.variant]

    models = {variant: build_model(variant) for variant in selected}
    if set(selected) != set(VARIANTS):
        fairness_models = {variant: build_model(variant) for variant in VARIANTS}
    else:
        fairness_models = models

    variant_results: dict[str, Any] = {}
    inventories: dict[str, Any] = {}
    transfers: dict[str, Any] = {}
    for variant, model in models.items():
        forward = forward_signature(
            model,
            imgsz=args.imgsz,
            batch=args.batch,
            device="cpu",
        )
        statistics = model_statistics(model, imgsz=args.imgsz)
        inventory = module_inventory(model.model)
        inventories[variant] = inventory
        transfers[variant] = transfer_weights(model, args.weights, apply=False)
        class_names = [module.__class__.__name__ for module in model.model.modules()]
        forbidden = [
            name
            for name in class_names
            if any(token in name.lower() for token in FORBIDDEN_TOKENS)
        ]
        gamma_keys = [
            key for key in model.model.state_dict() if ".gamma" in key
        ]
        gamma_values = [
            value
            for key, value in model.model.state_dict().items()
            if ".gamma" in key
        ]
        expected_sizes = [
            [args.imgsz // 8, args.imgsz // 8],
            [args.imgsz // 16, args.imgsz // 16],
            [args.imgsz // 32, args.imgsz // 32],
        ]
        checks = {
            "forward": True,
            "detect_has_three_inputs": len(forward["detect_input_shapes"]) == 3,
            "detect_channels": [
                shape[1] for shape in forward["detect_input_shapes"]
            ]
            == [64, 128, 256],
            "detect_shapes": forward["detect_spatial_sizes"] == expected_sizes,
            "detect_strides": forward["detect_strides"] == [8.0, 16.0, 32.0],
            "finite": forward["all_finite"],
            "no_forbidden_modules": not forbidden,
            "residual_gammas_in_state_dict": variant == "a_base" or bool(gamma_keys),
            "residual_gammas_start_at_point_one": variant == "a_base"
            or all(
                bool(
                    torch.allclose(
                        value.float(),
                        torch.full_like(value.float(), 0.1),
                    )
                )
                for value in gamma_values
            ),
            "gflops_profiled_or_error_recorded": (
                statistics["gflops"] is not None
                or bool(statistics["gflops_error"])
            ),
        }
        variant_results[variant] = {
            "yaml": VARIANTS[variant]["yaml"],
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "forward": forward,
            "statistics": statistics,
            "forbidden_modules": forbidden,
            "gamma_state_keys": gamma_keys,
        }
        print(
            json.dumps(
                {
                    "variant": variant,
                    "checks": checks,
                    "parameters": statistics["parameters"],
                    "gflops": statistics["gflops"],
                },
                ensure_ascii=False,
            )
        )

    equivalence = baseline_equivalence()
    fairness = backbone_fairness(fairness_models)
    window = window_padding_check()
    window_passed = all(
        (
            window["partition_reverse_exact"],
            window["nonstandard_attention_finite"],
            window["input_gradient_finite"],
            window["parameter_gradients_present"],
            window["positive_channel_temperature"],
        )
    )
    gradient_model = models.get("e_full") or build_model("e_full")
    rectangular = forward_signature(
        build_model("e_full"),
        imgsz=(args.imgsz, args.imgsz + 32),
        batch=1,
        device="cpu",
    )
    rectangular_passed = (
        rectangular["detect_spatial_sizes"]
        == [
            [args.imgsz // 8, (args.imgsz + 32) // 8],
            [args.imgsz // 16, (args.imgsz + 32) // 16],
            [args.imgsz // 32, (args.imgsz + 32) // 32],
        ]
        and rectangular["all_finite"]
    )
    gradients = gradient_check(
        gradient_model,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
    )
    gradient_passed = (
        gradients["all_trainable_parameters_have_gradients"]
        and gradients["all_gradients_finite"]
    )
    amp = cuda_amp_check("e_full", imgsz=args.imgsz)
    amp_passed = amp["status"] in {"passed", "not_run"}

    all_checks = (
        all(item["all_checks_passed"] for item in variant_results.values())
        and equivalence["all_checks_passed"]
        and fairness["all_checks_passed"]
        and window_passed
        and rectangular_passed
        and gradient_passed
        and amp_passed
    )
    summary = {
        "all_checks_passed": all_checks,
        "runtime": {
            **runtime_versions(),
            "platform": platform.platform(),
            "cpu_only_local_validation": not torch.cuda.is_available(),
        },
        "formal_training_started": False,
        "variants": variant_results,
        "baseline_equivalence": equivalence,
        "backbone_fairness": fairness,
        "window_padding_and_attention": window,
        "window_check_passed": window_passed,
        "rectangular_e_full_forward": rectangular,
        "rectangular_forward_passed": rectangular_passed,
        "e_full_backward": gradients,
        "gradient_check_passed": gradient_passed,
        "cuda_amp": amp,
        "weight_transfer_summary": {
            variant: {
                key: transfer[key]
                for key in (
                    "inherited_state_tensors",
                    "total_state_tensors",
                    "inherited_parameter_elements",
                    "target_parameter_elements",
                    "parameter_element_inheritance_ratio",
                    "backbone_parameter_inheritance_ratio",
                    "detect_parameter_inheritance_ratio",
                )
            }
            for variant, transfer in transfers.items()
        },
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "summary.json", summary)
    write_json(args.report_dir / "weight_transfer.json", transfers)
    write_json(args.report_dir / "module_inventory.json", inventories)

    base_stats = variant_results.get("a_base", {}).get("statistics")
    if base_stats is None:
        base_stats = model_statistics(fairness_models["a_base"], imgsz=args.imgsz)
    e_stats = variant_results.get("e_full", {}).get("statistics")
    if e_stats is None:
        e_stats = model_statistics(fairness_models["e_full"], imgsz=args.imgsz)
    rows = []
    for variant in VARIANTS:
        if variant in variant_results:
            stats = variant_results[variant]["statistics"]
        else:
            stats = model_statistics(fairness_models[variant], imgsz=args.imgsz)
        rows.append(
            {
                "variant": variant,
                "yaml": VARIANTS[variant]["yaml"],
                "parameters": stats["parameters"],
                "trainable_parameters": stats["trainable_parameters"],
                "gflops": stats["gflops"],
                "gflops_error": stats["gflops_error"],
                "delta_parameters_vs_a": stats["parameters"]
                - base_stats["parameters"],
                "delta_gflops_vs_a": (
                    stats["gflops"] - base_stats["gflops"]
                    if stats["gflops"] is not None
                    and base_stats["gflops"] is not None
                    else None
                ),
                "delta_parameters_vs_e": stats["parameters"] - e_stats["parameters"],
                "delta_gflops_vs_e": (
                    stats["gflops"] - e_stats["gflops"]
                    if stats["gflops"] is not None
                    and e_stats["gflops"] is not None
                    else None
                ),
            }
        )
    _write_comparison(args.report_dir / "model_comparison.csv", rows)
    _write_markdown(args.report_dir / "check_report.md", summary)
    print(f"Reports written to {args.report_dir.resolve()}")
    if not all_checks:
        raise SystemExit("ASCGD preflight failed; inspect reports/ascgd_preflight.")


if __name__ == "__main__":
    main()
