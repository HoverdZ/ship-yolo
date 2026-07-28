"""Build, transfer, forward, backward, and audit all six formal models on CPU."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from tools.paper_artifacts.formal_protocol import (
    EXPERIMENTS,
    FormalConfig,
    audit_model,
    build_and_initialize,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def run(weights: str | Path, imgsz: int = 64) -> dict:
    reports = {}
    with tempfile.TemporaryDirectory(prefix="formal-model-check-") as directory:
        temporary = Path(directory)
        for experiment_id in EXPERIMENTS:
            config = FormalConfig(
                experiment_id=experiment_id,
                local_runs_root=str(temporary / "runs"),
                local_data_root=str(temporary / experiment_id / "data"),
            )
            config.local_yaml.parent.mkdir(parents=True, exist_ok=True)
            config.local_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": str(temporary),
                        "train": "unused",
                        "val": "unused",
                        "test": "unused",
                        "nc": 1,
                        "names": {0: "ship"},
                    }
                ),
                encoding="utf-8",
            )
            model, transfer = build_and_initialize(config, weights)
            audit = audit_model(config, model, backward_imgsz=imgsz)
            reports[experiment_id] = {
                "strides": audit["strides"],
                "detect_from": audit["detect_from"],
                "loaded_tensors": transfer["loaded_tensors"],
                "total_tensors": transfer["target_state_tensors"],
                "tensor_ratio": transfer["tensor_inheritance_ratio"],
                "parameter_ratio": transfer["parameter_element_inheritance_ratio"],
                "passed": audit["passed"] and transfer["passed"],
            }
    return {"models": reports, "passed": all(item["passed"] for item in reports.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "yolo11n.pt"))
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--output", default="reports/formal_ablation_v1/model_check.json")
    args = parser.parse_args()
    report = run(args.weights, args.imgsz)
    write_json(ROOT / args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
