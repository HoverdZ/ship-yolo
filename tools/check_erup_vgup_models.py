"""CPU audit for the four ERUP/VGUP YOLO11n model configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.erup_vgup_utils import (
    EXPERIMENTS,
    backward_report,
    build_model,
    compatibility_report,
    forward_report,
    initialize_from_official,
    model_statistics,
    state_dict_roundtrip_report,
    structure_report,
    write_json,
)


def check_one(
    experiment: str,
    *,
    weights: str | Path,
    imgsz: int,
    print_network: bool,
) -> dict:
    model = build_model(experiment, verbose=print_network)
    if print_network:
        print(model.model)
    model.info(detailed=False, verbose=True, imgsz=imgsz)
    statistics = model_statistics(model, imgsz=imgsz)
    structure = structure_report(model, experiment)
    forward = forward_report(model, experiment, imgsz=imgsz)
    backward = backward_report(experiment, imgsz=64)
    checkpoint = state_dict_roundtrip_report(experiment, imgsz=64)
    inheritance = initialize_from_official(
        model,
        experiment,
        weights=weights,
        apply=True,
    )
    checks = {
        "parameters_positive": statistics["parameters"] > 0,
        "gflops_positive": statistics["gflops"] > 0,
        "structure": structure["passed"],
        "forward_640": forward["passed"],
        "backward": backward["passed"],
        "checkpoint": checkpoint["passed"],
        "inheritance": inheritance["passed"],
    }
    return {
        "experiment": experiment,
        "model_yaml": str(
            EXPERIMENTS[experiment]["yaml"].relative_to(ROOT)
        ).replace("\\", "/"),
        "statistics": statistics,
        "structure": structure,
        "forward": forward,
        "backward": backward,
        "checkpoint": checkpoint,
        "inheritance": inheritance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(EXPERIMENTS),
        action="append",
        dest="models",
    )
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--print-network", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/erup_vgup/erup_vgup_model_check.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.models or list(EXPERIMENTS)
    report = {"compatibility": compatibility_report(), "models": {}}
    for experiment in selected:
        print(f"\n[check] {experiment}")
        item = check_one(
            experiment,
            weights=args.weights,
            imgsz=args.imgsz,
            print_network=args.print_network,
        )
        report["models"][experiment] = item
        inheritance = item["inheritance"]
        print(
            f"  params={item['statistics']['parameters']:,} "
            f"GFLOPs={item['statistics']['gflops']:.4f} "
            f"detector={inheritance['detector_loaded_tensors']}/"
            f"{inheritance['detector_state_tensors']} "
            f"passed={item['passed']}"
        )
    report["passed"] = bool(
        report["compatibility"]["passed"]
        and all(item["passed"] for item in report["models"].values())
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("ERUP/VGUP model checks failed.")


if __name__ == "__main__":
    main()

