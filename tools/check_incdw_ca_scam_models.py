"""Build, backpropagate and audit both InceptionDW CA-SCAM models."""

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
from tools.incdw_ca_scam_experiments import (
    RUN_IDS,
    audit_incdw_ca_scam_topology,
    build_incdw_ca_scam_config,
)


def _size(model, imgsz: int) -> tuple[int, float]:
    from ultralytics.utils.torch_utils import get_flops

    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    return parameters, float(get_flops(model.model, imgsz=imgsz))


def check_models(*, backward_imgsz: int = 64) -> dict:
    """Return structure, CPU gradient, complexity and transfer audits."""

    register_custom_modules()
    controls = {
        "R01_PLS": ROOT
        / "experiments"
        / "formal_models"
        / "R01_yolo11n_pls.yaml",
        "R04_DPLS_CA_SCAM": ROOT
        / "experiments"
        / "formal_models"
        / "R04_yolo11n_dpls_ca_scam.yaml",
        "PLS_CA_SCAM_VGUP": ROOT
        / "experiments"
        / "pls_scam_family"
        / "PLS_CA_SCAM_VGUP_yolo11n.yaml",
    }
    reports: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(
        prefix="ship_yolo_incdw_ca_scam_"
    ) as temporary:
        control_sizes: dict[str, dict[str, float | int]] = {}
        for name, yaml_path in controls.items():
            model = formal._model_with_nc(yaml_path, nc=1)
            parameters, gflops = _size(model, 640)
            control_sizes[name] = {
                "parameters": parameters,
                "gflops_imgsz_640": gflops,
            }

        for run_id in RUN_IDS:
            config = replace(
                build_incdw_ca_scam_config(run_id, run_training=False),
                local_runs_root=temporary,
            )
            model = formal._model_with_nc(config.model_yaml, nc=1)
            topology = audit_incdw_ca_scam_topology(config, model)
            structure = formal.audit_model(
                config,
                model,
                backward_imgsz=backward_imgsz,
            )
            transfer = formal.transfer_official_weights(config, model)
            parameters, gflops = _size(model, 640)
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
                "gflops_imgsz_640": gflops,
                "passed": (
                    topology["passed"]
                    and structure["passed"]
                    and transfer["passed"]
                ),
            }
    return {
        "controls": control_sizes,
        "experiments": reports,
        "passed": len(reports) == len(RUN_IDS)
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
