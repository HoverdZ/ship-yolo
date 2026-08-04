"""Audit every unique formal model on CPU, including official weight transfer."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.formal_experiments.protocol import (
    FormalRunConfig,
    audit_model,
    build_and_initialize,
)
from tools.formal_experiments.registry import load_registry


def _weight_path(name: str, weights_dir: Path) -> str:
    candidate = weights_dir / name
    return str(candidate.resolve()) if candidate.is_file() else name


def run_check(
    *,
    backward_imgsz: int = 64,
    weights_dir: Path = ROOT,
) -> dict[str, Any]:
    registry = load_registry()
    reports: dict[str, Any] = {}
    checked_yamls: set[str] = set()
    with tempfile.TemporaryDirectory(
        prefix="formal-experiment-model-check-"
    ) as directory:
        temporary = Path(directory)
        data_yaml = temporary / "data.yaml"
        data_yaml.write_text(
            "\n".join(
                (
                    f"path: {temporary.as_posix()}",
                    "train: unused",
                    "val: unused",
                    "test: unused",
                    "nc: 1",
                    "names: {0: ship}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        for run_id, spec in registry["canonical_runs"].items():
            model_yaml = spec["model_yaml"]
            if model_yaml in checked_yamls:
                reports[run_id] = {
                    "reuses_model_yaml": next(
                        key
                        for key, value in reports.items()
                        if value.get("model_yaml") == model_yaml
                    ),
                    "model_yaml": model_yaml,
                    "passed": True,
                }
                continue
            checked_yamls.add(model_yaml)
            # Use a primary-data config only to avoid resolving the Colab-only
            # HRSC2016-MS runtime YAML during this local structure check.
            source_run = (
                "R00" if run_id == "S00" else "R10" if run_id == "S01" else run_id
            )
            config = FormalRunConfig.from_registry(source_run)
            config = replace(
                config,
                experiment_id=run_id,
                run_id=run_id,
                spec=dict(spec),
                model_yaml=ROOT / model_yaml,
                initialization_weight=_weight_path(
                    spec["initialization_weight"], weights_dir
                ),
                expected_detect_strides=tuple(
                    float(value) for value in spec["expected_detect_strides"]
                ),
                local_yaml=data_yaml,
                local_runs_root=str(temporary / "runs"),
                device="cpu",
            )
            model, transfer = build_and_initialize(config)
            structure = audit_model(
                config,
                model,
                backward_imgsz=backward_imgsz,
            )
            reports[run_id] = {
                "model_yaml": model_yaml,
                "parameters": structure["parameter_count"],
                "strides": structure["strides"],
                "loaded_tensors": transfer["loaded_tensors"],
                "total_tensors": transfer["target_state_tensors"],
                "loaded_total": transfer["loaded_total"],
                "parameter_element_inheritance_ratio": transfer[
                    "parameter_element_inheritance_ratio"
                ],
                "forward_backward": structure[
                    "cpu_forward_backward_checks"
                ],
                "passed": bool(transfer["passed"] and structure["passed"]),
            }
    return {
        "schema_version": 1,
        "unique_model_yamls": len(checked_yamls),
        "runs": reports,
        "passed": all(report["passed"] for report in reports.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward-imgsz", type=int, default=64)
    parser.add_argument("--weights-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "formal_experiments" / "model_check.json",
    )
    args = parser.parse_args()
    report = run_check(
        backward_imgsz=args.backward_imgsz,
        weights_dir=args.weights_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
