"""Generate auditable semantic weight-transfer reports for both FaPN models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fapn_utils import VARIANTS, build_model, semantic_weight_transfer, variant_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "inceptiondw", "all"], default="all")
    parser.add_argument("--weights", default="yolo11n.pt", help="Official YOLO11n checkpoint.")
    parser.add_argument("--apply", action="store_true", help="Also apply inherited tensors in memory.")
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for variant in variants:
        model = build_model(variant)
        report = semantic_weight_transfer(model, args.weights, apply=args.apply)
        report.update(
            {
                "variant": variant,
                "experiment_name": variant_config(variant)["name"],
                "model_yaml": str(variant_config(variant)["yaml"].relative_to(ROOT)).replace("\\", "/"),
            }
        )
        destination = variant_config(variant)["report"]
        write_json(destination, report)
        summary = {
            "variant": variant,
            "report": str(destination),
            "state_tensors": f"{report['inherited_state_tensors']}/{report['target_state_tensors']}",
            "parameter_tensors": f"{report['inherited_parameter_tensors']}/{report['target_parameter_tensors']}",
            "parameter_elements": f"{report['inherited_parameter_elements']}/{report['target_parameter_elements']}",
            "parameter_element_ratio": report["parameter_element_inheritance_ratio"],
            "shape_mismatches": len(report["shape_mismatches"]),
            "unmatched_target_keys": len(report["unmatched_target_keys"]),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
