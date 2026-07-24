"""Run pre-training structure, forward, SGTA, and weight-transfer audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.badc_scg_sgta_utils import VARIANTS, full_audit, write_json


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="reports/badc_scg_sgta")
    args = parser.parse_args(argv)

    variants = args.variant or list(VARIANTS)
    output = Path(args.output_dir).expanduser().resolve()
    reports = {
        variant: full_audit(variant, weights=args.weights, imgsz=args.imgsz)
        for variant in variants
    }
    for variant, report in reports.items():
        write_json(output / f"{variant}_audit.json", report)
        print(
            f"{variant}: passed={report['all_checks_passed']} "
            f"params={report['statistics']['parameters']} "
            f"GFLOPs={report['statistics']['gflops_at_imgsz']:.4f} "
            f"transfer={report['weight_transfer']['loaded_state_tensors']}/"
            f"{report['weight_transfer']['total_state_tensors']}"
        )
    summary = {
        "all_checks_passed": all(
            report["all_checks_passed"] for report in reports.values()
        ),
        "variants": reports,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({"all_checks_passed": summary["all_checks_passed"]}, indent=2))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)
    return reports


if __name__ == "__main__":
    main()
