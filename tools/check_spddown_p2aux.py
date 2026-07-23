"""Run CPU pre-training audits for SPDDown and P2 Gaussian auxiliary supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.spddown_p2aux_utils import full_audit, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("spddown", "p2_gaussian_aux", "all"),
        default="all",
    )
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="reports/spddown_p2aux")
    return parser


def main(argv: list[str] | None = None) -> dict[str, dict]:
    args = build_parser().parse_args(argv)
    variants = ("spddown", "p2_gaussian_aux") if args.variant == "all" else (args.variant,)
    reports = {}
    for variant in variants:
        report = full_audit(variant, weights=args.weights, imgsz=args.imgsz)
        output = Path(args.output_dir) / f"{variant}_audit.json"
        write_json(output, report)
        reports[variant] = report
        print(
            f"{variant}: passed={report['all_checks_passed']} "
            f"params={report['statistics']['parameters']} "
            f"GFLOPs={report['statistics']['gflops_at_imgsz']:.6f} "
            f"loaded={report['weight_transfer']['loaded_state_tensors']}/"
            f"{report['weight_transfer']['total_state_tensors']}"
        )
        if not report["all_checks_passed"]:
            raise RuntimeError(json.dumps(report["checks"], ensure_ascii=False))
    return reports


if __name__ == "__main__":
    main()
