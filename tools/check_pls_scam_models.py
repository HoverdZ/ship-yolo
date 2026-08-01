"""Build, backpropagate and audit all four PLS-SCAM formal models."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_custom_modules
from tools.formal_experiments import protocol as formal
from tools.pls_scam_experiments import (
    RUN_IDS,
    audit_pls_scam_topology,
    build_pls_scam_config,
)


def check_models(*, backward_imgsz: int = 64) -> dict:
    """Return structure, CPU gradient, complexity and transfer audits."""

    register_custom_modules()
    from ultralytics.utils.torch_utils import get_flops

    reports: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="ship_yolo_pls_scam_") as temporary:
        baseline_config = replace(
            build_pls_scam_config(RUN_IDS[0], run_training=False),
            local_runs_root=temporary,
        )
        baseline_yaml = (
            ROOT / "experiments" / "formal_models" / "R01_yolo11n_pls.yaml"
        )
        baseline = formal._model_with_nc(baseline_yaml, nc=1)
        baseline_parameters = sum(
            parameter.numel() for parameter in baseline.model.parameters()
        )
        baseline_gflops = float(get_flops(baseline.model, imgsz=640))

        for run_id in RUN_IDS:
            config = replace(
                build_pls_scam_config(run_id, run_training=False),
                local_runs_root=temporary,
            )
            model = formal._model_with_nc(config.model_yaml, nc=1)
            topology = audit_pls_scam_topology(config, model)
            structure = formal.audit_model(
                config,
                model,
                backward_imgsz=backward_imgsz,
            )
            transfer = formal.transfer_official_weights(config, model)
            parameters = sum(
                parameter.numel() for parameter in model.model.parameters()
            )
            gflops = float(get_flops(model.model, imgsz=640))
            reports[run_id] = {
                "topology": topology,
                "structure": structure,
                "official_transfer": {
                    "loaded_total": transfer["loaded_total"],
                    "loaded_tensors": transfer["loaded_tensors"],
                    "target_state_tensors": transfer["target_state_tensors"],
                    "parameter_element_inheritance_ratio": transfer[
                        "parameter_element_inheritance_ratio"
                    ],
                    "passed": transfer["passed"],
                },
                "parameters": parameters,
                "parameter_increase_over_r01": parameters - baseline_parameters,
                "gflops_imgsz_640": gflops,
                "gflops_increase_over_r01": gflops - baseline_gflops,
                "passed": (
                    topology["passed"]
                    and structure["passed"]
                    and transfer["passed"]
                ),
            }
    return {
        "baseline_r01": {
            "parameters": baseline_parameters,
            "gflops_imgsz_640": baseline_gflops,
        },
        "experiments": reports,
        "passed": len(reports) == 4
        and all(report["passed"] for report in reports.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward-imgsz", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_models(backward_imgsz=args.backward_imgsz)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
