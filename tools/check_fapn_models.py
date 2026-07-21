"""Run CPU build, forward, backward, topology, stats, and transfer checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fapn_utils import VARIANTS, full_check, variant_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "inceptiondw", "all"], default="all")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="", help="Optional combined JSON report.")
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    reports = {}
    for variant in variants:
        report = full_check(variant, args.weights, imgsz=args.imgsz)
        reports[variant] = report
        transfer_report = dict(report["weight_transfer"])
        transfer_report.update(
            {
                "variant": variant,
                "experiment_name": variant_config(variant)["name"],
                "model_yaml": str(variant_config(variant)["yaml"].relative_to(ROOT)).replace("\\", "/"),
            }
        )
        write_json(variant_config(variant)["report"], transfer_report)
        print(f"\n[{variant}] FaPN node shapes")
        for index, shapes in report["forward"]["fapn_nodes"].items():
            print(f"  layer {index}: input={shapes['input']} -> output={shapes['output']}")
        print(f"  Detect inputs: {report['forward']['detect_input_shapes']}")
        print(f"  Parameters: {report['statistics']['parameters']}")
        print(f"  GFLOPs: {report['statistics']['gflops']:.6f}")
        print(
            "  Inherited parameter elements: "
            f"{report['weight_transfer']['inherited_parameter_elements']}/"
            f"{report['weight_transfer']['target_parameter_elements']} "
            f"({report['weight_transfer']['parameter_element_inheritance_ratio']:.4%})"
        )
        print(f"  Checks passed: {report['all_checks_passed']}")

    combined = {
        "all_checks_passed": all(report["all_checks_passed"] for report in reports.values()),
        "reports": reports,
    }
    if args.output:
        write_json(args.output, combined)
    if not combined["all_checks_passed"]:
        print(json.dumps({variant: report["checks"] for variant, report in reports.items()}, indent=2))
        raise SystemExit("One or more FaPN checks failed.")


if __name__ == "__main__":
    main()
