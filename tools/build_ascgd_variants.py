"""Build and profile ASCGD variants without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ascgd_utils import (
    REPORT_DIR,
    VARIANTS,
    build_model,
    model_statistics,
    module_inventory,
    variant_config,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "build_summary.json",
    )
    args = parser.parse_args()
    selected = list(VARIANTS) if args.variant == "all" else [args.variant]
    payload = {}
    for variant in selected:
        model = build_model(variant)
        payload[variant] = {
            "yaml": str(variant_config(variant)["yaml_path"]),
            "statistics": model_statistics(model, imgsz=args.imgsz),
            "detect_strides": [float(value) for value in model.model.stride.tolist()],
            "module_inventory": module_inventory(model.model),
        }
        print(
            json.dumps(
                {
                    "variant": variant,
                    "statistics": payload[variant]["statistics"],
                    "detect_strides": payload[variant]["detect_strides"],
                },
                ensure_ascii=False,
            )
        )
    write_json(args.output, payload)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
