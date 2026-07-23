"""Fast structural tests for the ASCGD experiment suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.ascgd import (
    ASCGDChannelP4,
    ASCGDChannelP5,
    ASCGDDirectP3,
    ASCGDDirectP4,
    ASCGDDirectP5,
    ASCGDGather,
    ASCGDSpatialP3,
    ASCGDSwappedP3,
    ASCGDSwappedP4,
    ASCGDSwappedP5,
    ASCGDSymmetricP3,
    ASCGDSymmetricP4,
    ASCGDSymmetricP5,
    ChannelCrossAttention,
    WindowCrossAttention,
    window_partition,
    window_reverse,
)
from tools.ascgd_utils import (
    DEFAULT_WEIGHTS,
    VARIANTS,
    backbone_fairness,
    baseline_equivalence,
    build_model,
    forward_signature,
    gradient_check,
    transfer_weights,
)


@pytest.fixture(scope="session")
def models():
    return {variant: build_model(variant) for variant in VARIANTS}


def test_all_variants_build_and_keep_three_scales(models) -> None:
    for model in models.values():
        signature = forward_signature(model, imgsz=128, batch=2)
        assert signature["detect_spatial_sizes"] == [[16, 16], [8, 8], [4, 4]]
        assert signature["detect_strides"] == [8.0, 16.0, 32.0]
        assert signature["all_finite"]


def test_e_full_handles_rectangular_non_window_multiple_features() -> None:
    signature = forward_signature(
        build_model("e_full"),
        imgsz=(128, 160),
        batch=1,
    )
    assert signature["detect_spatial_sizes"] == [[16, 20], [8, 10], [4, 5]]
    assert signature["all_finite"]


def test_a_is_exact_validated_inceptiondw_baseline() -> None:
    report = baseline_equivalence()
    assert report["all_checks_passed"], report


def test_every_variant_has_identical_backbone(models) -> None:
    report = backbone_fairness(models)
    assert report["all_checks_passed"], report


@pytest.mark.parametrize("height,width", [(40, 40), (37, 45), (41, 32)])
def test_window_partition_reverse_exact(height: int, width: int) -> None:
    x = torch.randn(2, 128, height, width)
    windows, metadata = window_partition(x, 8)
    restored = window_reverse(windows, metadata, 8)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)


def test_attention_nonstandard_shape_backward_and_temperature() -> None:
    x = torch.randn(2, 128, 17, 19, requires_grad=True)
    spatial = WindowCrossAttention(128, 128)
    channel = ChannelCrossAttention(128, 128)
    output = spatial(x, x) + channel(x, x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.all(channel.positive_temperature > 0)
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None
        for parameter in list(spatial.parameters()) + list(channel.parameters())
        if parameter.requires_grad
    )


def test_variant_distribution_modules_are_explicit(models) -> None:
    expected = {
        "b_gather": (ASCGDDirectP3, ASCGDDirectP4, ASCGDDirectP5),
        "c_sca": (ASCGDSpatialP3, ASCGDDirectP4, ASCGDDirectP5),
        "d_cca": (ASCGDDirectP3, ASCGDChannelP4, ASCGDChannelP5),
        "e_full": (ASCGDSpatialP3, ASCGDChannelP4, ASCGDChannelP5),
        "f_swap": (ASCGDSwappedP3, ASCGDSwappedP4, ASCGDSwappedP5),
        "g_symmetric": (
            ASCGDSymmetricP3,
            ASCGDSymmetricP4,
            ASCGDSymmetricP5,
        ),
    }
    for variant, distribution_types in expected.items():
        top = list(models[variant].model.model)
        assert isinstance(top[14], ASCGDGather)
        assert isinstance(top[15], distribution_types[0])
        assert isinstance(top[17], distribution_types[1])
        assert isinstance(top[19], distribution_types[2])


def test_e_full_backward_reaches_every_trainable_parameter() -> None:
    report = gradient_check(build_model("e_full"), imgsz=64, batch=2)
    assert report["all_trainable_parameters_have_gradients"], report[
        "missing_gradient_keys"
    ]
    assert report["all_gradients_finite"], report["nonfinite_gradient_keys"]


def test_official_weight_transfer_maps_backbone_and_shifted_detect() -> None:
    report = transfer_weights(
        build_model("e_full"),
        DEFAULT_WEIGHTS,
        apply=False,
    )
    assert report["source_detect_index"] == 23
    assert report["target_detect_index"] == 21
    assert report["backbone_parameter_inheritance_ratio"] > 0.99
    assert 0.0 < report["detect_parameter_inheritance_ratio"] < 1.0
    assert report["shape_mismatches"]
    assert not report["forced_crop_repeat_or_pad"]


@pytest.mark.parametrize(
    "script",
    [
        "build_ascgd_variants.py",
        "check_ascgd.py",
        "init_ascgd_weights.py",
        "train_ascgd_colab.py",
        "validate_ascgd.py",
        "benchmark_ascgd.py",
        "summarize_ascgd_results.py",
    ],
)
def test_tool_help_does_not_start_work(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
