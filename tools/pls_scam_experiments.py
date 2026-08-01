"""Controlled formal experiments combining PLS with SCAM-family modules."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from torch import nn

from tools.formal_experiments import protocol as formal
from tools.formal_experiments.registry import ROOT, load_yaml
from tools.paper_artifacts.formal_protocol import write_json

REGISTRY_PATH = ROOT / "experiments" / "pls_scam_family" / "registry.yaml"
RUN_IDS = (
    "PLS_CA_SCAM_150ep",
    "PLS_SCAM_150ep",
    "PLS_CA_SCAM_VGUP_150ep",
    "PLS_CA_SCAM_ERUP_150ep",
)
MODEL_NAMES = {
    "PLS_CA_SCAM_150ep": "YOLO11n + PLS + CA-SCAM",
    "PLS_SCAM_150ep": "YOLO11n + PLS + SCAM",
    "PLS_CA_SCAM_VGUP_150ep": "YOLO11n + PLS + CA-SCAM + VGUP",
    "PLS_CA_SCAM_ERUP_150ep": "YOLO11n + PLS + CA-SCAM + ERUP",
}
FROZEN_TRAINING = {
    "imgsz": 640,
    "epochs": 150,
    "batch": 8,
    "workers": 2,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "patience": 100,
    "seed": 0,
    "cache": "disk",
    "deterministic": False,
    "save_period": 10,
    "plots": True,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "scale": 0.5,
    "translate": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}


def load_pls_scam_registry() -> dict[str, Any]:
    """Load and validate the four independent experiment definitions."""

    registry = load_yaml(REGISTRY_PATH)
    experiments = registry.get("experiments")
    if not isinstance(experiments, dict) or tuple(experiments) != RUN_IDS:
        raise ValueError(
            "PLS-SCAM registry order/content must match the four fixed runs."
        )
    shared = registry.get("shared", {})
    expected_shared = {
        "initialization_weight": "yolo11n.pt",
        "epochs": 150,
        "imgsz": 640,
        "batch": 8,
        "seed": 0,
        "upsampling": "nearest",
        "expected_detect_strides": [4, 8, 16],
    }
    for key, value in expected_shared.items():
        if shared.get(key) != value:
            raise ValueError(f"Unexpected shared {key}: {shared.get(key)!r}")
    for run_id, experiment in experiments.items():
        model_yaml = ROOT / str(experiment.get("model_yaml", ""))
        if not model_yaml.is_file():
            raise FileNotFoundError(f"{run_id}: {model_yaml}")
        if not str(experiment.get("notebook_path", "")).startswith(
            "notebooks/formal/"
        ):
            raise ValueError(f"{run_id}: notebook must be under notebooks/formal")
    return registry


def build_pls_scam_config(
    run_id: str,
    *,
    run_training: bool = True,
) -> formal.FormalRunConfig:
    """Reuse all frozen R01 settings and change only declared modules."""

    registry = load_pls_scam_registry()
    if run_id not in registry["experiments"]:
        raise KeyError(f"Unknown PLS-SCAM experiment: {run_id}")
    experiment = registry["experiments"][run_id]
    modules = dict(experiment["modules"])
    base = formal.FormalRunConfig.from_registry(
        "R01",
        run_training=run_training,
    )
    training = dict(base.training)
    for key, expected in FROZEN_TRAINING.items():
        if training.get(key) != expected:
            raise RuntimeError(
                f"Frozen R01 protocol drifted at {key}: "
                f"{training.get(key)!r} != {expected!r}"
            )
    module_flags = {
        "pls": True,
        "dpls": False,
        "scam": bool(modules["scam"]),
        "ca_scam": modules["ca_scam"],
        "erup": bool(modules["erup"]),
        "vgup": bool(modules["vgup"]),
    }
    if module_flags["vgup"]:
        module_flags.update(
            {
                "global_gate": True,
                "spatial_gate": True,
            }
        )
    spec = {
        "experiment_id": run_id,
        "canonical_run_id": run_id,
        "paper_aliases": [run_id.removesuffix("_150ep")],
        "base_model": MODEL_NAMES[run_id],
        "model_yaml": experiment["model_yaml"],
        "initialization_weight": "yolo11n.pt",
        "initialization_kind": "official_yolo11n",
        "dataset_id": "primary_ship_v1",
        "data_yaml": base.drive_data_yaml,
        "imgsz": 640,
        "epochs": 150,
        "batch": 8,
        "optimizer": training["optimizer"],
        "seed": 0,
        "module_flags": module_flags,
        "expected_detect_strides": [4, 8, 16],
        "output_drive_dir": experiment["output_drive_dir"],
        "notebook_path": experiment["notebook_path"],
        "status": "prepared_not_run",
        "artifact_manifest": (
            f"formal_experiments/{run_id}/seed_0/run_manifest.json"
        ),
        "description": experiment["description"],
    }
    return replace(
        base,
        experiment_id=run_id,
        run_id=run_id,
        seed=0,
        model_yaml=ROOT / str(experiment["model_yaml"]),
        initialization_weight="yolo11n.pt",
        expected_detect_strides=(4.0, 8.0, 16.0),
        spec=spec,
        training=training,
        run_training=bool(run_training),
        run_test_evaluation=False,
        # T4/L4/A100 are all allowed; the exact runtime is still recorded.
        enforce_environment_lock=False,
    )


def _upsample_is_pls(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Upsample)
        and module.mode == "nearest"
        and float(module.scale_factor) == 2.0
    )


def audit_pls_scam_topology(
    config: formal.FormalRunConfig,
    model,
) -> dict[str, Any]:
    """Verify PLS, attention placement, preprocessors, PAN and Detect."""

    layers = model.model.model
    flags = config.spec["module_flags"]
    shifted = bool(flags.get("vgup") or flags.get("erup"))
    offset = 1 if shifted else 0
    attention_name = "SCAM" if flags.get("scam") else "CASCAM"
    attention_layers = [21 + offset, 22 + offset, 23 + offset]
    attention_sources = [14 + offset, 17 + offset, 20 + offset]
    detect_layer = 24 + offset
    checks = {
        "first_upsample_is_nearest_x2": _upsample_is_pls(layers[9 + offset]),
        "second_upsample_is_nearest_x2": _upsample_is_pls(layers[12 + offset]),
        "top_down_first_concat_unchanged": (
            type(layers[10 + offset]).__name__ == "Concat"
            and list(layers[10 + offset].f) == [-1, 4 + offset]
        ),
        "top_down_second_concat_unchanged": (
            type(layers[13 + offset]).__name__ == "Concat"
            and list(layers[13 + offset].f) == [-1, 2 + offset]
        ),
        "pan_first_concat_unchanged": (
            type(layers[16 + offset]).__name__ == "Concat"
            and list(layers[16 + offset].f) == [-1, 11 + offset]
        ),
        "pan_second_concat_unchanged": (
            type(layers[19 + offset]).__name__ == "Concat"
            and list(layers[19 + offset].f) == [-1, 8 + offset]
        ),
        "three_expected_attention_modules": all(
            type(layers[index]).__name__ == attention_name
            for index in attention_layers
        ),
        "attention_sources_are_p2_p3_p4": [
            int(layers[index].f) for index in attention_layers
        ]
        == attention_sources,
        "detect_layer_index": detect_layer == len(layers) - 1,
        "detect_uses_attention_outputs": list(layers[detect_layer].f)
        == attention_layers,
        "detect_strides_are_p2_p3_p4": [
            float(value) for value in model.model.stride
        ]
        == [4.0, 8.0, 16.0],
        "no_dysample": all(type(layer).__name__ != "DySample" for layer in layers),
    }
    expected_preprocessor = None
    if flags.get("vgup"):
        expected_preprocessor = "VGUPPreprocessor"
        checks["vgup_is_first_layer"] = type(layers[0]).__name__ == expected_preprocessor
        checks["vgup_global_gate_enabled"] = layers[0].use_global_gate is True
        checks["vgup_spatial_gate_enabled"] = layers[0].use_spatial_gate is True
    elif flags.get("erup"):
        expected_preprocessor = "ERUPPreprocessor"
        checks["erup_is_first_layer"] = type(layers[0]).__name__ == expected_preprocessor
    else:
        checks["no_input_preprocessor"] = type(layers[0]).__name__ == "Conv"

    report = {
        "run_id": config.run_id,
        "upsample_layers": [9 + offset, 12 + offset],
        "attention_type": attention_name,
        "attention_layers": attention_layers,
        "attention_sources": attention_sources,
        "input_preprocessor": expected_preprocessor,
        "pan_concat_layers": [16 + offset, 19 + offset],
        "detect_layer": detect_layer,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        raise AssertionError(f"PLS-SCAM topology audit failed: {report}")
    return report


def prepare_pls_scam_experiment(
    config: formal.FormalRunConfig,
) -> dict[str, Any]:
    """Run complete data, transfer, structure and CPU audits before training."""

    prepared = formal.prepare_experiment(config)
    topology = audit_pls_scam_topology(config, prepared["model"])
    write_json(
        config.protocol_staging_dir / "pls_scam_topology_audit.json",
        topology,
    )
    prepared["pls_scam_topology"] = topology
    return prepared


def update_pls_scam_comparison(config: formal.FormalRunConfig) -> Path:
    """Collect only completed real metrics for R01 and the four new runs."""

    formal_root = Path(config.drive_project_root) / "formal_experiments"
    output = (
        Path(config.drive_project_root)
        / "paper_artifacts"
        / "tables"
        / "pls_scam_family_comparison.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "experiment_id",
        "initialization",
        "epochs",
        "loaded_total",
        "precision",
        "recall",
        "map50",
        "map75",
        "map50_95",
        "best_epoch",
    )
    rows: list[dict[str, Any]] = []
    for run_id in ("R01", *RUN_IDS):
        run_dir = formal_root / run_id / "seed_0"
        metrics_path = run_dir / "validation_metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest_path = run_dir / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        transfer = manifest.get("official_transfer") or {}
        best = manifest.get("best_epoch") or {}
        rows.append(
            {
                "experiment_id": run_id,
                "initialization": "official yolo11n.pt",
                "epochs": 150,
                "loaded_total": transfer.get("loaded_total", ""),
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "map50": metrics.get("map50", ""),
                "map75": metrics.get("map75", ""),
                "map50_95": metrics.get("map50_95", ""),
                "best_epoch": best.get("best_epoch", ""),
            }
        )
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return output


__all__ = [
    "FROZEN_TRAINING",
    "MODEL_NAMES",
    "RUN_IDS",
    "audit_pls_scam_topology",
    "build_pls_scam_config",
    "load_pls_scam_registry",
    "prepare_pls_scam_experiment",
    "update_pls_scam_comparison",
]
