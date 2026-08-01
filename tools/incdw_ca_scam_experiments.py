"""Formal InceptionDW experiments with controlled PLS/DPLS CA-SCAM heads."""

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

REGISTRY_PATH = (
    ROOT / "experiments" / "incdw_ca_scam_family" / "registry.yaml"
)
RUN_IDS = (
    "INCDW_PLS_CA_SCAM_VGUP_150ep",
    "INCDW_DPLS_CA_SCAM_150ep",
)
MODEL_NAMES = {
    "INCDW_PLS_CA_SCAM_VGUP_150ep": (
        "YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP"
    ),
    "INCDW_DPLS_CA_SCAM_150ep": (
        "YOLO11n + InceptionDW + DPLS + CA-SCAM"
    ),
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


def load_incdw_ca_scam_registry() -> dict[str, Any]:
    """Load and validate the two independent experiment definitions."""

    registry = load_yaml(REGISTRY_PATH)
    experiments = registry.get("experiments")
    if not isinstance(experiments, dict) or tuple(experiments) != RUN_IDS:
        raise ValueError(
            "InceptionDW CA-SCAM registry must contain the two fixed runs."
        )
    expected_shared = {
        "initialization_weight": "yolo11n.pt",
        "epochs": 150,
        "imgsz": 640,
        "batch": 8,
        "seed": 0,
        "inceptiondw_scope": "p2_p3_bottleneck_cv2_only",
        "ca_scam": "complete_bounded",
        "expected_detect_strides": [4, 8, 16],
    }
    shared = registry.get("shared", {})
    for key, expected in expected_shared.items():
        if shared.get(key) != expected:
            raise ValueError(f"Unexpected shared {key}: {shared.get(key)!r}")

    model_paths: list[str] = []
    notebook_paths: list[str] = []
    output_paths: list[str] = []
    for run_id, experiment in experiments.items():
        modules = experiment.get("modules", {})
        if modules.get("inceptiondw") is not True:
            raise ValueError(f"{run_id}: InceptionDW must be enabled.")
        if modules.get("ca_scam") != "bounded":
            raise ValueError(f"{run_id}: bounded CA-SCAM is required.")
        if bool(modules.get("pls")) == bool(modules.get("dpls")):
            raise ValueError(f"{run_id}: exactly one of PLS/DPLS is required.")
        model_path = str(experiment.get("model_yaml", ""))
        notebook_path = str(experiment.get("notebook_path", ""))
        output_path = str(experiment.get("output_drive_dir", ""))
        if not (ROOT / model_path).is_file():
            raise FileNotFoundError(f"{run_id}: {ROOT / model_path}")
        if not notebook_path.startswith("notebooks/formal/"):
            raise ValueError(f"{run_id}: Notebook must be under notebooks/formal.")
        model_paths.append(model_path)
        notebook_paths.append(notebook_path)
        output_paths.append(output_path)
    if len(set(model_paths)) != len(RUN_IDS):
        raise ValueError("Every run requires an independent model YAML.")
    if len(set(notebook_paths)) != len(RUN_IDS):
        raise ValueError("Every run requires an independent Notebook.")
    if len(set(output_paths)) != len(RUN_IDS):
        raise ValueError("Every run requires an independent Drive output.")
    return registry


def build_incdw_ca_scam_config(
    run_id: str,
    *,
    run_training: bool = True,
) -> formal.FormalRunConfig:
    """Reuse the frozen R01 protocol and change only declared modules."""

    registry = load_incdw_ca_scam_registry()
    if run_id not in registry["experiments"]:
        raise KeyError(f"Unknown InceptionDW CA-SCAM experiment: {run_id}")
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
        "inceptiondw": True,
        "pls": bool(modules["pls"]),
        "dpls": bool(modules["dpls"]),
        "scam": False,
        "ca_scam": "bounded",
        "erup": False,
        "vgup": bool(modules["vgup"]),
    }
    if module_flags["vgup"]:
        module_flags.update({"global_gate": True, "spatial_gate": True})

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
        enforce_environment_lock=False,
    )


def _is_pls(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Upsample)
        and module.mode == "nearest"
        and float(module.scale_factor) == 2.0
    )


def _is_dpls(module: nn.Module) -> bool:
    return (
        type(module).__name__ == "DySample"
        and module.scale == 2
        and module.style == "lp"
        and module.groups == 4
        and module.dyscope is False
    )


def audit_incdw_ca_scam_topology(
    config: formal.FormalRunConfig,
    model,
) -> dict[str, Any]:
    """Verify scoped InceptionDW, upsampling, CA-SCAM, PAN and Detect."""

    layers = model.model.model
    flags = config.spec["module_flags"]
    shifted = bool(flags.get("vgup"))
    offset = 1 if shifted else 0
    inception_layers = [2 + offset, 4 + offset]
    p4_c3k2_layer = 6 + offset
    upsample_layers = [9 + offset, 12 + offset]
    attention_layers = [21 + offset, 22 + offset, 23 + offset]
    attention_sources = [14 + offset, 17 + offset, 20 + offset]
    detect_layer = 24 + offset

    bottleneck_cv1_preserved = True
    bottleneck_cv2_replaced = True
    for index in inception_layers:
        block = layers[index]
        bottlenecks = list(block.m)
        bottleneck_cv1_preserved &= bool(bottlenecks) and all(
            type(item.cv1).__name__ == "Conv" for item in bottlenecks
        )
        bottleneck_cv2_replaced &= bool(bottlenecks) and all(
            type(item.cv2).__name__ == "InceptionDWConvBNAct"
            for item in bottlenecks
        )

    checks = {
        "exactly_two_p2_p3_inceptiondw_blocks": (
            [type(layer).__name__ for layer in layers].count(
                "C3k2_InceptionDW"
            )
            == 2
            and all(
                type(layers[index]).__name__ == "C3k2_InceptionDW"
                for index in inception_layers
            )
        ),
        "bottleneck_first_conv_preserved": bottleneck_cv1_preserved,
        "bottleneck_second_conv_is_inceptiondw": bottleneck_cv2_replaced,
        "p4_backbone_uses_official_c3k2": (
            type(layers[p4_c3k2_layer]).__name__ == "C3k2"
        ),
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
        "three_bounded_ca_scam_modules": all(
            type(layers[index]).__name__ == "CASCAM"
            for index in attention_layers
        ),
        "ca_scam_sources_are_p2_p3_p4": [
            int(layers[index].f) for index in attention_layers
        ]
        == attention_sources,
        "detect_layer_index": detect_layer == len(layers) - 1,
        "detect_uses_ca_scam_outputs": list(layers[detect_layer].f)
        == attention_layers,
        "detect_strides_are_p2_p3_p4": [
            float(value) for value in model.model.stride
        ]
        == [4.0, 8.0, 16.0],
    }

    upsampling = "PLS" if flags["pls"] else "DPLS"
    if flags["pls"]:
        checks["two_nearest_pls_nodes"] = all(
            _is_pls(layers[index]) for index in upsample_layers
        )
        checks["no_dysample"] = all(
            type(layer).__name__ != "DySample" for layer in layers
        )
    else:
        checks["two_official_dysample_nodes"] = all(
            _is_dpls(layers[index]) for index in upsample_layers
        )
        checks["exactly_two_dysample_nodes"] = (
            [type(layer).__name__ for layer in layers].count("DySample") == 2
        )

    preprocessor = None
    if flags["vgup"]:
        preprocessor = "VGUPPreprocessor"
        checks["vgup_is_first_layer"] = (
            type(layers[0]).__name__ == preprocessor
        )
        checks["vgup_global_gate_enabled"] = layers[0].use_global_gate is True
        checks["vgup_spatial_gate_enabled"] = layers[0].use_spatial_gate is True
    else:
        checks["no_input_preprocessor"] = type(layers[0]).__name__ == "Conv"

    report = {
        "run_id": config.run_id,
        "inceptiondw_layers": inception_layers,
        "inceptiondw_scope": "P2/P3 C3k2 bottleneck cv2 only",
        "upsampling": upsampling,
        "upsample_layers": upsample_layers,
        "attention_type": "CASCAM",
        "attention_layers": attention_layers,
        "attention_sources": attention_sources,
        "input_preprocessor": preprocessor,
        "pan_concat_layers": [16 + offset, 19 + offset],
        "detect_layer": detect_layer,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        raise AssertionError(f"InceptionDW CA-SCAM audit failed: {report}")
    return report


def prepare_incdw_ca_scam_experiment(
    config: formal.FormalRunConfig,
) -> dict[str, Any]:
    """Run complete data, transfer, structure and CPU audits before training."""

    prepared = formal.prepare_experiment(config)
    topology = audit_incdw_ca_scam_topology(config, prepared["model"])
    write_json(
        config.protocol_staging_dir / "incdw_ca_scam_topology_audit.json",
        topology,
    )
    prepared["incdw_ca_scam_topology"] = topology
    return prepared


def update_incdw_ca_scam_comparison(config: formal.FormalRunConfig) -> Path:
    """Collect completed real metrics for the new runs and their controls."""

    formal_root = Path(config.drive_project_root) / "formal_experiments"
    output = (
        Path(config.drive_project_root)
        / "paper_artifacts"
        / "tables"
        / "incdw_ca_scam_comparison.csv"
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
    controls = (
        "R01",
        "R04",
        "PLS_CA_SCAM_VGUP_150ep",
        *RUN_IDS,
    )
    for run_id in controls:
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
    "audit_incdw_ca_scam_topology",
    "build_incdw_ca_scam_config",
    "load_incdw_ca_scam_registry",
    "prepare_incdw_ca_scam_experiment",
    "update_incdw_ca_scam_comparison",
]
