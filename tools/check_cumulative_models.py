"""CPU validation for the three cumulative YOLO11n model configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cumulative_models_utils import (
    EXPERIMENTS,
    backward_report,
    build_model,
    compatibility_report,
    forward_report,
    model_statistics,
    state_dict_roundtrip_report,
    structure_report,
    transfer_pretrained_weights,
    write_json,
)


def check_one(
    experiment: str,
    *,
    weights: str | Path,
    imgsz: int,
    try_1024: bool,
    print_network: bool,
) -> dict:
    model = build_model(experiment, verbose=print_network)
    if print_network:
        print(model.model)
    model.info(detailed=False, verbose=True, imgsz=imgsz)

    stats = model_statistics(model, imgsz=imgsz)
    structure = structure_report(model, experiment)
    forward = forward_report(model, experiment, imgsz=imgsz)
    backward = backward_report(experiment)
    roundtrip = state_dict_roundtrip_report(experiment)
    transfer = transfer_pretrained_weights(
        model,
        weights,
        apply=True,
    )

    optional_1024 = {
        "attempted": try_1024,
        "passed": None,
        "error": None,
    }
    if try_1024:
        try:
            report_1024 = forward_report(
                build_model(experiment),
                experiment,
                imgsz=1024,
            )
            optional_1024.update(
                {
                    "passed": report_1024["passed"],
                    "detect_input_shapes": report_1024[
                        "detect_input_shapes"
                    ],
                }
            )
        except (MemoryError, RuntimeError) as error:
            optional_1024["passed"] = False
            optional_1024["error"] = (
                f"{type(error).__name__}: {error}"
            )

    checks = {
        "parameters_positive": stats["parameters"] > 0,
        "gflops_positive": stats["gflops"] > 0,
        "structure": structure["passed"],
        "forward_640": forward["passed"],
        "backward": backward["passed"],
        "state_dict_roundtrip": roundtrip["passed"],
        "weight_transfer": transfer["passed"],
    }
    return {
        "experiment": experiment,
        "model_yaml": str(
            EXPERIMENTS[experiment]["yaml"].relative_to(ROOT)
        ).replace("\\", "/"),
        "statistics": stats,
        "structure": structure,
        "forward": forward,
        "backward": backward,
        "state_dict_roundtrip": roundtrip,
        "weight_transfer": transfer,
        "optional_1024_forward": optional_1024,
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
    parser.add_argument("--try-1024", action="store_true")
    parser.add_argument("--print-network", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports/cumulative_models/cumulative_model_check.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.models or list(EXPERIMENTS)
    report = {
        "compatibility": compatibility_report(),
        "models": {},
    }
    for experiment in selected:
        print(f"\n[check] {experiment}")
        item = check_one(
            experiment,
            weights=args.weights,
            imgsz=args.imgsz,
            try_1024=args.try_1024,
            print_network=args.print_network,
        )
        report["models"][experiment] = item
        print(
            f"  params={item['statistics']['parameters']:,} "
            f"GFLOPs={item['statistics']['gflops']:.4f} "
            f"transfer={item['weight_transfer']['inherited_tensors']}/"
            f"{item['weight_transfer']['target_state_tensors']} "
            f"passed={item['passed']}"
        )

    report["passed"] = bool(
        report["compatibility"]["passed"]
        and all(item["passed"] for item in report["models"].values())
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("Cumulative model checks failed.")


if __name__ == "__main__":
    main()
