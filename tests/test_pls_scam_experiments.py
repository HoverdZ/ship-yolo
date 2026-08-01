from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from custom_modules.register import register_custom_modules
from tools.formal_experiments import protocol as formal
from tools.pls_scam_experiments import (
    FROZEN_TRAINING,
    RUN_IDS,
    audit_pls_scam_topology,
    build_pls_scam_config,
    load_pls_scam_registry,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_configs_reuse_the_frozen_r01_protocol(run_id: str) -> None:
    config = build_pls_scam_config(run_id, run_training=True)
    for key, value in FROZEN_TRAINING.items():
        assert config.training[key] == value
    assert config.initialization_weight == "yolo11n.pt"
    assert config.expected_detect_strides == (4.0, 8.0, 16.0)
    assert config.copy_workers == 32
    assert config.run_training is True
    assert config.run_test_evaluation is False
    assert config.enforce_environment_lock is False
    assert config.spec["module_flags"]["dpls"] is False


def test_registry_defines_exactly_four_independent_runs() -> None:
    registry = load_pls_scam_registry()
    assert tuple(registry["experiments"]) == RUN_IDS
    paths = [
        experiment["model_yaml"]
        for experiment in registry["experiments"].values()
    ]
    notebooks = [
        experiment["notebook_path"]
        for experiment in registry["experiments"].values()
    ]
    outputs = [
        experiment["output_drive_dir"]
        for experiment in registry["experiments"].values()
    ]
    assert len(paths) == len(set(paths)) == 4
    assert len(notebooks) == len(set(notebooks)) == 4
    assert len(outputs) == len(set(outputs)) == 4


@pytest.mark.parametrize(
    ("target_name", "source_name"),
    (
        ("PLS_SCAM_yolo11n.yaml", "R03_yolo11n_dpls_scam.yaml"),
        ("PLS_CA_SCAM_yolo11n.yaml", "R04_yolo11n_dpls_ca_scam.yaml"),
        (
            "PLS_CA_SCAM_VGUP_yolo11n.yaml",
            "R10_yolo11n_dpls_ca_scam_vgup.yaml",
        ),
        (
            "PLS_CA_SCAM_ERUP_yolo11n.yaml",
            "R06_yolo11n_dpls_ca_scam_erup.yaml",
        ),
    ),
)
def test_each_yaml_only_replaces_two_dysample_layers_with_pls(
    target_name: str,
    source_name: str,
) -> None:
    target = yaml.safe_load(
        (ROOT / "experiments/pls_scam_family" / target_name).read_text(
            encoding="utf-8"
        )
    )
    source = yaml.safe_load(
        (ROOT / "experiments/formal_models" / source_name).read_text(
            encoding="utf-8"
        )
    )
    assert target["backbone"] == source["backbone"]
    assert len(target["head"]) == len(source["head"])
    changed = []
    for index, (target_layer, source_layer) in enumerate(
        zip(target["head"], source["head"])
    ):
        if target_layer != source_layer:
            changed.append(index)
            assert source_layer[2] == "DySample"
            assert target_layer == [-1, 1, "nn.Upsample", [None, 2, "nearest"]]
    assert changed == [0, 3]


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_models_build_with_expected_pls_attention_topology(run_id: str) -> None:
    register_custom_modules()
    from ultralytics import YOLO

    config = build_pls_scam_config(run_id, run_training=False)
    model = YOLO(str(config.model_yaml), verbose=False)
    report = audit_pls_scam_topology(config, model)
    assert report["passed"]
    assert report["checks"]["no_dysample"]


def test_weight_mapping_targets_match_new_layer_indices() -> None:
    assert formal._detect_mapping("PLS_SCAM_150ep")[0] == 24
    assert formal._detect_mapping("PLS_CA_SCAM_150ep")[0] == 24
    assert formal._detect_mapping("PLS_CA_SCAM_VGUP_150ep")[0] == 25
    assert formal._detect_mapping("PLS_CA_SCAM_ERUP_150ep")[0] == 25
    vgup_map = formal._layer_mapping("PLS_CA_SCAM_VGUP_150ep")
    erup_map = formal._layer_mapping("PLS_CA_SCAM_ERUP_150ep")
    assert vgup_map[1] == 0 and vgup_map[16] == 17
    assert erup_map[1] == 0 and erup_map[16] == 17
