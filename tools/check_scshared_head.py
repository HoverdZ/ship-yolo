"""Run the SCSharedHead structure, forward, compute, and transfer preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scshared_head_utils import full_audit, write_json


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="reports/scshared_head")
    args = parser.parse_args(argv)

    report = full_audit(weights=args.weights, imgsz=args.imgsz)
    output = Path(args.output_dir).expanduser().resolve()
    write_json(output / "scshared_head_audit.json", report)
    write_json(
        output / "summary.json",
        {
            "all_checks_passed": report["all_checks_passed"],
            "experiment_name": report["experiment_name"],
            "statistics": report["statistics"],
            "weight_transfer": {
                "loaded_state_tensors": report["weight_transfer"][
                    "loaded_state_tensors"
                ],
                "total_state_tensors": report["weight_transfer"][
                    "total_state_tensors"
                ],
                "loaded_target_parameter_element_ratio": report[
                    "weight_transfer"
                ]["loaded_target_parameter_element_ratio"],
            },
        },
    )
    print(
        f"passed={report['all_checks_passed']} "
        f"params={report['statistics']['parameters']} "
        f"GFLOPs={report['statistics']['gflops_at_imgsz']:.4f} "
        f"transfer={report['weight_transfer']['loaded_state_tensors']}/"
        f"{report['weight_transfer']['total_state_tensors']} "
        f"mapped_head={report['weight_transfer']['mapped_native_detect_output_tensors']}"
    )
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
