"""Build, transfer, and CPU forward/backward audit all convolution screens."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.conv_screening_utils import (  # noqa: E402
    ConvScreeningConfig,
    EXPERIMENTS,
    prepare_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "yolo11n.pt",
        help="Official YOLO11n pretrained checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    reports: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="ship_yolo_conv_screening_") as temporary:
        temporary_root = Path(temporary)
        for experiment_id in EXPERIMENTS:
            local_root = temporary_root / experiment_id / "data"
            config = ConvScreeningConfig(
                experiment_id=experiment_id,
                local_data_root=str(local_root),
                drive_runs_root=str(temporary_root / "runs"),
            )
            config.local_yaml.parent.mkdir(parents=True, exist_ok=True)
            config.local_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(local_root),
                        "train": "images/train",
                        "val": "images/val",
                        "names": {0: "ship"},
                        "nc": 1,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            prepared = prepare_model(
                config,
                args.weights,
                run_cpu_check=True,
            )
            reports[experiment_id] = {
                "model_yaml": str(config.model_yaml.relative_to(ROOT)),
                "selected_module": config.spec["module"],
                "structure": prepared["structure"],
                "weight_transfer": {
                    key: value
                    for key, value in prepared["transfer"].items()
                    if key not in {
                        "loaded_target_keys",
                        "unmatched_target_keys",
                        "p2_p3_cv1_expected_keys",
                    }
                },
                "cpu_smoke": prepared["cpu_smoke"],
                "passed": (
                    prepared["structure"]["passed"]
                    and prepared["transfer"]["passed"]
                    and prepared["cpu_smoke"]["passed"]
                ),
            }
            print(
                experiment_id,
                f"Loaded/Total={prepared['transfer']['loaded_total']}",
                f"CPU={prepared['cpu_smoke']['passed']}",
            )
    report = {
        "ultralytics": "8.4.92",
        "official_weights": str(args.weights),
        "experiments": reports,
        "passed": all(item["passed"] for item in reports.values()),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print("Wrote:", args.output)
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
