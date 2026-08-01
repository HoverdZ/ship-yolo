from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from custom_modules.register import register_custom_modules
from tools.formal_experiments import protocol as formal
from tools.incdw_ca_scam_experiments import (
    FROZEN_TRAINING,
    RUN_IDS,
    audit_incdw_ca_scam_topology,
    build_incdw_ca_scam_config,
    load_incdw_ca_scam_registry,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_configs_reuse_frozen_formal_protocol(run_id: str) -> None:
    config = build_incdw_ca_scam_config(run_id, run_training=True)
    for key, value in FROZEN_TRAINING.items():
        assert config.training[key] == value
    assert config.initialization_weight == "yolo11n.pt"
    assert config.expected_detect_strides == (4.0, 8.0, 16.0)
    assert config.copy_workers == 32
    assert config.run_training is True
    assert config.run_test_evaluation is False
    assert config.enforce_environment_lock is False
    assert config.spec["module_flags"]["inceptiondw"] is True
    assert config.spec["module_flags"]["ca_scam"] == "bounded"


def test_registry_defines_two_independent_runs() -> None:
    registry = load_incdw_ca_scam_registry()
    assert tuple(registry["experiments"]) == RUN_IDS
    paths = [item["model_yaml"] for item in registry["experiments"].values()]
    notebooks = [
        item["notebook_path"] for item in registry["experiments"].values()
    ]
    outputs = [
        item["output_drive_dir"] for item in registry["experiments"].values()
    ]
    assert len(paths) == len(set(paths)) == 2
    assert len(notebooks) == len(set(notebooks)) == 2
    assert len(outputs) == len(set(outputs)) == 2


@pytest.mark.parametrize(
    ("target_name", "source_path", "changed_indices"),
    (
        (
            "INCDW_PLS_CA_SCAM_VGUP_yolo11n.yaml",
            "experiments/pls_scam_family/PLS_CA_SCAM_VGUP_yolo11n.yaml",
            [3, 5],
        ),
        (
            "INCDW_DPLS_CA_SCAM_yolo11n.yaml",
            "experiments/formal_models/R04_yolo11n_dpls_ca_scam.yaml",
            [2, 4],
        ),
    ),
)
def test_yaml_only_replaces_p2_p3_c3k2_with_inceptiondw(
    target_name: str,
    source_path: str,
    changed_indices: list[int],
) -> None:
    target = yaml.safe_load(
        (
            ROOT / "experiments" / "incdw_ca_scam_family" / target_name
        ).read_text(encoding="utf-8")
    )
    source = yaml.safe_load((ROOT / source_path).read_text(encoding="utf-8"))
    assert target["head"] == source["head"]
    assert len(target["backbone"]) == len(source["backbone"])
    changes = []
    for index, (target_layer, source_layer) in enumerate(
        zip(target["backbone"], source["backbone"])
    ):
        if target_layer != source_layer:
            changes.append(index)
            assert source_layer[2] == "C3k2"
            assert target_layer[2] == "C3k2_InceptionDW"
            assert target_layer[:2] == source_layer[:2]
            assert target_layer[3] == source_layer[3]
    assert changes == changed_indices


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_models_build_with_expected_inceptiondw_topology(run_id: str) -> None:
    register_custom_modules()
    from ultralytics import YOLO

    config = build_incdw_ca_scam_config(run_id, run_training=False)
    model = YOLO(str(config.model_yaml), verbose=False)
    report = audit_incdw_ca_scam_topology(config, model)
    assert report["passed"]
    assert report["checks"]["bottleneck_first_conv_preserved"]
    assert report["checks"]["bottleneck_second_conv_is_inceptiondw"]
    assert report["checks"]["p4_backbone_uses_official_c3k2"]


def test_weight_mapping_matches_new_layer_indices() -> None:
    assert formal._detect_mapping("INCDW_DPLS_CA_SCAM_150ep")[0] == 24
    assert formal._detect_mapping("INCDW_PLS_CA_SCAM_VGUP_150ep")[0] == 25
    vgup_map = formal._layer_mapping("INCDW_PLS_CA_SCAM_VGUP_150ep")
    dpls_map = formal._layer_mapping("INCDW_DPLS_CA_SCAM_150ep")
    assert vgup_map[1] == 0 and vgup_map[7] == 6
    assert dpls_map[0] == 0 and dpls_map[6] == 6
