"""Unit and integration tests for the official FaPN port."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.fapn import (
    FaPNAlign,
    FaPNFeatureSelection,
    FaPNModulatedDeformConv2d,
)
from tools.fapn_training import audit_dataset, copy_dataset_to_local
from tools.fapn_utils import (
    build_model,
    forward_report,
    semantic_weight_transfer,
    structure_report,
    topology_report,
)


def test_fsm_shape_and_official_residual_formula() -> None:
    torch.manual_seed(0)
    module = FaPNFeatureSelection(16, 8).eval()
    x = torch.randn(2, 16, 13, 11)
    with torch.inference_mode():
        attention = torch.sigmoid(module.conv_attention(F.adaptive_avg_pool2d(x, 1)))
        expected = module.projection(x + x * attention)
        actual = module(x)
    assert actual.shape == (2, 8, 13, 11)
    torch.testing.assert_close(actual, expected)
    assert isinstance(module.sigmoid, nn.Sigmoid)
    assert module.conv_attention.bias is None
    assert module.projection.bias is None


@pytest.mark.parametrize(
    ("lateral_shape", "topdown_shape"),
    [((1, 16, 20, 20), (1, 8, 20, 20)), ((1, 16, 20, 20), (1, 8, 10, 10))],
)
def test_fapn_align_same_and_double_size_inputs(lateral_shape, topdown_shape) -> None:
    module = FaPNAlign(16, 8, 8).eval()
    feat_l = torch.randn(*lateral_shape)
    feat_s = torch.randn(*topdown_shape)
    with torch.inference_mode():
        output = module([feat_l, feat_s])
    assert output.shape == (1, 8, 20, 20)
    assert torch.isfinite(output).all()


def test_fapn_align_keeps_feat_up_times_two_for_offset_source() -> None:
    module = FaPNAlign(16, 8, 8).eval()
    feat_l = torch.randn(1, 16, 12, 12)
    feat_s = torch.randn(1, 8, 6, 6)
    captured: list[torch.Tensor] = []
    hook = module.offset_feature.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    try:
        with torch.inference_mode():
            module([feat_l, feat_s])
            feat_arm = module.fsm(feat_l)
            feat_up = F.interpolate(feat_s, size=(12, 12), mode="bilinear", align_corners=False)
    finally:
        hook.remove()
    expected = torch.cat((feat_arm, feat_up * 2.0), dim=1)
    torch.testing.assert_close(captured[0], expected)


def test_dcnv2_offset_mask_channels_and_zero_initialization() -> None:
    module = FaPNModulatedDeformConv2d(8)
    assert module.deformable_groups == 8
    assert module.offset_channels == 144
    assert module.mask_channels == 72
    assert module.conv_offset_mask.out_channels == 216
    assert torch.count_nonzero(module.dcn.bias) == 0
    assert torch.count_nonzero(module.conv_offset_mask.weight) == 0
    assert torch.count_nonzero(module.conv_offset_mask.bias) == 0
    feature = torch.randn(1, 8, 9, 7)
    offset, mask = module.offset_and_mask(feature)
    assert offset.shape == (1, 144, 9, 7)
    assert mask.shape == (1, 72, 9, 7)
    torch.testing.assert_close(offset, torch.zeros_like(offset))
    torch.testing.assert_close(mask, torch.full_like(mask, 0.5))


def test_dcnv2_rejects_non_divisible_channels() -> None:
    with pytest.raises(ValueError, match="divisible"):
        FaPNModulatedDeformConv2d(12)


def test_fapn_gradient_backpropagation() -> None:
    module = FaPNAlign(16, 8, 8).train()
    feat_l = torch.randn(1, 16, 16, 16, requires_grad=True)
    feat_s = torch.randn(1, 8, 8, 8, requires_grad=True)
    loss = module([feat_l, feat_s]).square().mean()
    loss.backward()
    assert feat_l.grad is not None and torch.isfinite(feat_l.grad).all()
    assert feat_s.grad is not None and torch.isfinite(feat_s.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_yaml_topology_and_m4_recursion_are_exact() -> None:
    report = topology_report()
    assert report["all_checks_passed"], report


@pytest.mark.parametrize("variant", ["baseline", "inceptiondw"])
def test_both_models_build_with_unchanged_pan(variant: str) -> None:
    model = build_model(variant)
    report = structure_report(model, variant)
    assert report["all_checks_passed"], report
    assert report["pan"]["stride_conv_indices"] == [16, 19]
    assert report["pan"]["concat_indices"] == [17, 20]
    assert report["pan"]["c3k2_indices"] == [18, 21]
    assert report["pan"]["second_concat_from"] == [-1, 10]


@pytest.mark.parametrize("variant", ["baseline", "inceptiondw"])
def test_both_models_forward_with_three_scales(variant: str) -> None:
    report = forward_report(build_model(variant), imgsz=64)
    assert report["all_checks_passed"], report
    assert report["detect_input_shapes"] == [
        [1, 64, 8, 8],
        [1, 128, 4, 4],
        [1, 256, 2, 2],
    ]


@pytest.mark.parametrize("variant", ["baseline", "inceptiondw"])
def test_semantic_weight_transfer_never_uses_shifted_raw_names(variant: str) -> None:
    report = semantic_weight_transfer(build_model(variant), ROOT / "yolo11n.pt")
    assert report["mapping_strategy"] == "explicit semantic layer mapping"
    assert report["source_to_target_layer_map"]["17"] == 16
    assert report["source_to_target_layer_map"]["23"] == 22
    assert report["category_report"]["fapn"]["inherited_parameter_elements"] == 0
    assert report["category_report"]["pan"]["element_ratio"] > 0.97
    assert report["parameter_element_inheritance_ratio"] > 0.80


def test_dataset_copy_and_train_val_test_count_audit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val", "test"):
        images = source / "images" / split
        labels = source / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / "sample.jpg").write_bytes(b"test")
        (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data = {
        "path": str(source),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "ship"},
    }
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")
    local_yaml = copy_dataset_to_local(data_yaml, tmp_path / "local")
    report = audit_dataset(local_yaml)
    for split in ("train", "val", "test"):
        assert report["splits"][split]["images"] == 1
        assert report["splits"][split]["labels"] == 1
        assert report["splits"][split]["counts_equal"]


@pytest.mark.parametrize(
    "script",
    [
        "tools/check_fapn_models.py",
        "tools/transfer_fapn_weights.py",
        "tools/train_yolo11n_fapn.py",
        "tools/train_yolo11n_inceptiondw_fapn.py",
    ],
)
def test_cli_help(script: str) -> None:
    result = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_training_workflow_does_not_launch_training_subprocess() -> None:
    import tools.fapn_training as training

    source = inspect.getsource(training)
    assert "import subprocess" not in source
    assert ".train(" in source
