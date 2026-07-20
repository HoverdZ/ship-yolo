"""Regression tests for the isolated backbone SCConv experiment."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import ultralytics
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.c3k2_scconv import C3k2_SCConv
from custom_modules.scconv import CRU, SRU, ScConv
from tools.scconv_utils import MODEL_YAML


@pytest.mark.parametrize("channels", [16, 32, 64, 128, 256])
def test_scconv_preserves_shape_and_is_finite(channels: int) -> None:
    module = ScConv(channels)
    sample = torch.randn(1, channels, 8, 8)
    output = module(sample)
    assert output.shape == sample.shape
    assert torch.isfinite(output).all()


def test_scconv_backward_reaches_all_parameters() -> None:
    module = ScConv(32)
    output = module(torch.randn(1, 32, 8, 8))
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_scconv_supports_paper_c1_to_c2_form() -> None:
    module = ScConv(32, out_channels=16)
    output = module(torch.randn(1, 32, 8, 8))
    assert output.shape == (1, 16, 8, 8)
    assert torch.isfinite(output).all()


def test_scconv_custom_group_normalization_path() -> None:
    module = ScConv(32, torch_gn=False)
    output = module(torch.randn(1, 32, 8, 8))
    output.mean().backward()
    assert output.shape == (1, 32, 8, 8)
    assert all(parameter.grad is not None for parameter in module.parameters())


@pytest.mark.parametrize("c3k", [False, True])
def test_c3k2_scconv_modes_preserve_interface(c3k: bool) -> None:
    module = C3k2_SCConv(64, 128, n=2, c3k=c3k, e=0.5, shortcut=True)
    output = module(torch.randn(1, 64, 12, 12))
    assert output.shape == (1, 128, 12, 12)


def test_invalid_channel_configurations_fail_clearly() -> None:
    with pytest.raises(ValueError, match="even channel"):
        SRU(15)
    with pytest.raises(ValueError, match="group_size"):
        CRU(18, group_size=4)
    with pytest.raises(ValueError, match="zero channels"):
        CRU(2, squeeze_ratio=2)


def test_yaml_changes_only_backbone_c3k2_module_names() -> None:
    config = yaml.safe_load(MODEL_YAML.read_text(encoding="utf-8"))
    baseline_path = (
        Path(ultralytics.__file__).resolve().parent
        / "cfg"
        / "models"
        / "11"
        / "yolo11.yaml"
    )
    expected = copy.deepcopy(
        yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    )
    for layer_index in (2, 4, 6, 8):
        expected["backbone"][layer_index][2] = "C3k2_SCConv"
    assert config == expected
    assert [layer[2] for layer in config["backbone"]].count("C3k2_SCConv") == 4
    assert [layer[2] for layer in config["head"]].count("C3k2_SCConv") == 0
    assert [layer[2] for layer in config["head"]].count("C3k2") == 4
    assert config["head"][-1][0] == [16, 19, 22]


def test_dynamic_registration_builds_four_backbone_nodes() -> None:
    from custom_modules.register import register_scconv_modules

    register_scconv_modules()
    from ultralytics import YOLO

    wrapper = YOLO(str(MODEL_YAML))
    nodes = [
        module for module in wrapper.model.model if isinstance(module, C3k2_SCConv)
    ]
    assert [int(module.i) for module in nodes] == [2, 4, 6, 8]
