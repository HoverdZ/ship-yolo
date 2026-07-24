"""Unit and model-build tests for CrossConv, DD, and CGFM ablations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from ultralytics.nn.modules import Bottleneck, C3x

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.c3k2_crossconv import C3k2CrossConv
from custom_modules.cgfm import AlignConcat, CGFM
from custom_modules.dd import DD, DPL
from tools.module_ablation_utils import (
    CONTROL_EXPERIMENT,
    EXPERIMENTS,
    build_model,
    structure_report,
    transfer_pretrained_weights,
)


@pytest.mark.parametrize("c3k", [False, True])
@pytest.mark.parametrize("shortcut", [False, True])
def test_c3k2_crossconv_forward_backward_and_variants(c3k: bool, shortcut: bool) -> None:
    module = C3k2CrossConv(24, 40, n=2, c3k=c3k, e=0.5, shortcut=shortcut)
    assert len(module.m) == 2
    assert all(isinstance(unit, C3x if c3k else Bottleneck) for unit in module.m)
    if not c3k:
        for unit in module.m:
            assert unit.cv1.conv.kernel_size == (1, 3)
            assert unit.cv2.conv.kernel_size == (3, 1)
            assert unit.add is shortcut
    x = torch.randn(2, 24, 17, 19, requires_grad=True)
    output = module(x)
    assert output.shape == (2, 40, 17, 19)
    output.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_c3k2_crossconv_nonmatching_channels_disable_unit_shortcut() -> None:
    unit = Bottleneck(12, 16, shortcut=True, k=((1, 3), (3, 1)), e=1.0)
    assert not unit.add
    assert unit(torch.randn(1, 12, 9, 11)).shape == (1, 16, 9, 11)


@pytest.mark.parametrize(
    ("shape", "c1", "c2"),
    [
        ((2, 16, 20, 24), 16, 24),
        ((2, 16, 21, 25), 16, 32),
        ((1, 24, 31, 28), 24, 16),
    ],
)
def test_dd_even_odd_channels_branches_and_backward(
    shape: tuple[int, ...],
    c1: int,
    c2: int,
) -> None:
    module = DD(c1, c2, r=4)
    x = torch.randn(*shape, requires_grad=True)
    conv_output, dpl_output = module.branch_outputs(x)
    expected_hw = ((shape[-2] + 1) // 2, (shape[-1] + 1) // 2)
    assert conv_output.shape == dpl_output.shape
    assert conv_output.shape == (shape[0], c2, *expected_hw)
    output = module(x)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dpl_uses_one_shared_extract_instance() -> None:
    dpl = DPL(16, 24, r=4)
    extract_modules = [module for module in dpl.modules() if module.__class__.__name__ == "Extract"]
    assert len(extract_modules) == 1


def test_cgfm_shape_weight_shape_order_and_backward() -> None:
    module = CGFM(c_deep=48, c_shallow=24, reduction=16)
    deep = torch.randn(2, 48, 13, 17, requires_grad=True)
    shallow = torch.randn(2, 24, 13, 17, requires_grad=True)
    aligned = module.align(deep)
    weights = module.channel_weights(torch.cat((aligned, shallow), dim=1))
    output = module([deep, shallow])
    assert weights.shape == (2, 24, 1, 1)
    assert output.shape == (2, 48, 13, 17)
    output.mean().backward()
    assert deep.grad is not None and shallow.grad is not None
    assert torch.isfinite(deep.grad).all() and torch.isfinite(shallow.grad).all()


def test_cgfm_and_control_reject_spatial_mismatch() -> None:
    inputs = [torch.randn(1, 32, 10, 10), torch.randn(1, 16, 9, 10)]
    with pytest.raises(ValueError, match="identical spatial sizes"):
        CGFM(32, 16)(inputs)
    with pytest.raises(ValueError, match="identical spatial sizes"):
        AlignConcat(32, 16)(inputs)


@pytest.mark.parametrize(
    ("experiment_name", "model_yaml"),
    list(EXPERIMENTS.items()) + list(CONTROL_EXPERIMENT.items()),
)
def test_model_yaml_builds_exact_replacement_scope(
    experiment_name: str,
    model_yaml: Path,
) -> None:
    model = build_model(model_yaml)
    report = structure_report(model, experiment_name)
    assert report["passed"], report
    assert isinstance(model.model.model[-1], nn.Module)


@pytest.mark.parametrize(("experiment_name", "model_yaml"), list(EXPERIMENTS.items()))
def test_pretrained_transfer_is_partial_and_shape_safe(
    experiment_name: str,
    model_yaml: Path,
) -> None:
    model = build_model(model_yaml)
    report = transfer_pretrained_weights(model, ROOT / "yolo11n.pt", apply=True)
    assert 0 < report["matched_tensors"] < report["target_tensors"], experiment_name
    assert report["unmatched_target_tensors"] == len(report["unmatched_target_keys"])
    assert 0.75 < report["tensor_match_ratio"] < 1.0


def test_no_inceptiondw_crossconv_combination_yaml_exists() -> None:
    assert not (ROOT / "experiments/yolo11n-inceptiondw-c3cross.yaml").exists()
