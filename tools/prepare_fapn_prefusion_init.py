"""Create the real pretrained FaPN-Prefusion init.pt files and manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fapn_prefusion_utils import VARIANTS, prepare_initialization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "inceptiondw", "all"], default="all")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    summary = {}
    for variant in variants:
        result = prepare_initialization(
            variant,
            weights=args.weights,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        transfer = result["weight_transfer"]
        summary[variant] = {
            "init_pt": result["init_pt"],
            "manifest": result["manifest"],
            "transfer_report": result["transfer_report"],
            "inherited_parameter_elements": transfer["inherited_parameter_elements"],
            "target_parameter_elements": transfer["target_parameter_elements"],
            "inheritance_ratio": transfer["parameter_element_inheritance_ratio"],
        }
        print(
            f"{variant}: inherited {transfer['inherited_parameter_elements']}/"
            f"{transfer['target_parameter_elements']} parameter elements; "
            f"saved {result['init_pt']}"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
