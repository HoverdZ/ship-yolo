"""Run the required local CPU checks for the CA-SCAM experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.calibrated_scam_utils import (
    backward_report,
    build_model,
    forward_report,
    initialize_from_official,
    model_statistics,
    state_dict_roundtrip_report,
    structure_report,
    write_json,
)
from tools.erup_vgup_utils import (
    build_model as build_base_model,
    model_statistics as base_model_statistics,
)


def run_checks(
    *,
    weights: str = "yolo11n.pt",
    full_imgsz: int = 640,
) -> dict:
    model = build_model(verbose=False)
    base = build_base_model("incdw_dysample_sfl_scam_vgup", verbose=False)
    inheritance = initialize_from_official(model, weights=weights, apply=True)
    ca_stats = model_statistics(model, imgsz=full_imgsz)
    base_stats = base_model_statistics(base, imgsz=full_imgsz)
    report = {
        "structure": structure_report(model),
        "inheritance": inheritance,
        "statistics": {
            "base": base_stats,
            "ca_scam": ca_stats,
            "parameter_delta": ca_stats["parameters"] - base_stats["parameters"],
        },
        "forward_640": forward_report(model, imgsz=full_imgsz),
        "backward": backward_report(imgsz=64),
        "roundtrip": state_dict_roundtrip_report(imgsz=64),
    }
    report["passed"] = (
        report["structure"]["passed"]
        and report["inheritance"]["passed"]
        and report["statistics"]["parameter_delta"] == 33
        and report["forward_640"]["passed"]
        and report["backward"]["passed"]
        and report["roundtrip"]["passed"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output",
        default="artifacts/ca_scam_check_report.json",
    )
    args = parser.parse_args()
    report = run_checks(weights=args.weights, full_imgsz=args.imgsz)
    write_json(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("CA-SCAM checks failed.")


if __name__ == "__main__":
    main()
