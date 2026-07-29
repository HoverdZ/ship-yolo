"""Build and inspect D0/D1/D2 without weights, training, or YOLO validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_custom_modules

SPECS = {
    "D0": {
        "yaml": ROOT / "experiments" / "formal_ablation_v1" / "A0_yolo11n.yaml",
        "strides": [8.0, 16.0, 32.0],
        "detect_from": [16, 19, 22],
        "nearest": 2,
        "dysample": 0,
    },
    "D1": {
        "yaml": ROOT / "experiments" / "dpls_controlled" / "D1_yolo11n_pls_nearest.yaml",
        "strides": [4.0, 8.0, 16.0],
        "detect_from": [14, 17, 20],
        "nearest": 2,
        "dysample": 0,
    },
    "D2": {
        "yaml": ROOT / "experiments" / "dpls_controlled" / "D2_yolo11n_pls_dysample.yaml",
        "strides": [4.0, 8.0, 16.0],
        "detect_from": [14, 17, 20],
        "nearest": 0,
        "dysample": 2,
    },
}
BANNED = ("C3k2_InceptionDW", "SCAM", "CASCAM", "VGUPPreprocessor", "ERUPPreprocessor")


def _backbone_signature(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.dumps(payload["backbone"], ensure_ascii=False, separators=(",", ":"))


def check() -> dict:
    from ultralytics import YOLO

    register_custom_modules()
    reports = {}
    for experiment_id, spec in SPECS.items():
        yaml_text = spec["yaml"].read_text(encoding="utf-8")
        for banned in BANNED:
            if banned in yaml_text:
                raise AssertionError(f"{experiment_id} contains banned module {banned}.")
        model = YOLO(str(spec["yaml"]), task="detect")
        module_names = [module.__class__.__name__ for module in model.model.modules()]
        strides = [float(value) for value in model.model.stride]
        detect_from = list(model.model.model[-1].f)
        nearest_count = module_names.count("Upsample")
        dysample_count = module_names.count("DySample")
        if strides != spec["strides"]:
            raise AssertionError(f"{experiment_id} strides: {strides} != {spec['strides']}")
        if detect_from != spec["detect_from"]:
            raise AssertionError(f"{experiment_id} Detect sources: {detect_from} != {spec['detect_from']}")
        if nearest_count != spec["nearest"] or dysample_count != spec["dysample"]:
            raise AssertionError(
                f"{experiment_id} upsamplers: nearest={nearest_count}, DySample={dysample_count}"
            )
        if any(name in BANNED for name in module_names):
            raise AssertionError(f"{experiment_id} instantiated a banned module.")
        reports[experiment_id] = {
            "yaml": spec["yaml"].relative_to(ROOT).as_posix(),
            "strides": strides,
            "detect_from": detect_from,
            "nearest_modules": nearest_count,
            "dysample_modules": dysample_count,
            "parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
            "passed": True,
        }
    if _backbone_signature(SPECS["D1"]["yaml"]) != _backbone_signature(SPECS["D2"]["yaml"]):
        raise AssertionError("D1 and D2 backbones differ.")
    return {"models": reports, "d1_d2_backbones_identical": True, "training_started": False, "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = check()
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
