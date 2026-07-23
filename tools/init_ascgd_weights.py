"""Create auditable ASCGD initialization checkpoints from official YOLO11n."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ascgd_utils import (
    DEFAULT_WEIGHTS,
    REPORT_DIR,
    VARIANTS,
    save_initialized_model,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="e_full")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--init-from-inception-best",
        type=Path,
        help="Debug only: use a trained InceptionDW checkpoint instead of official yolo11n.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "ascgd_init",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_DIR / "weight_transfer.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = args.init_from_inception_best or args.weights
    source_is_inception = args.init_from_inception_best is not None
    if source_is_inception:
        print(
            "WARNING: --init-from-inception-best is a debug-only path and is not "
            "the formal ablation initialization."
        )
    selected = list(VARIANTS) if args.variant == "all" else [args.variant]
    reports = {}
    for variant in selected:
        output = args.output_dir / f"yolo11n_incdw_ascgd_{variant}_init.pt"
        reports[variant] = save_initialized_model(
            variant,
            source,
            output,
            source_is_inception=source_is_inception,
            seed=args.seed,
        )
        transfer = reports[variant]["transfer"]
        print(
            json.dumps(
                {
                    "variant": variant,
                    "output": str(output.resolve()),
                    "inherited_parameter_elements": transfer[
                        "inherited_parameter_elements"
                    ],
                    "target_parameter_elements": transfer[
                        "target_parameter_elements"
                    ],
                    "inheritance_ratio": transfer[
                        "parameter_element_inheritance_ratio"
                    ],
                    "backbone_ratio": transfer[
                        "backbone_parameter_inheritance_ratio"
                    ],
                    "detect_ratio": transfer["detect_parameter_inheritance_ratio"],
                },
                ensure_ascii=False,
            )
        )
    write_json(args.report, reports)
    print(f"Wrote {args.report.resolve()}")


if __name__ == "__main__":
    main()
