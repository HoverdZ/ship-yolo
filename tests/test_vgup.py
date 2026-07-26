"""Unit tests for the complete, non-ablated VGUP module."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.vgup import (
    DepthwiseSeparableConv,
    LightweightVGUPEncoder,
    VGUPPreprocessor,
)


def test_vgup_contains_all_three_fixed_improvements() -> None:
    module = VGUPPreprocessor()
    assert isinstance(module.encoder, LightweightVGUPEncoder)
    blocks = [
        item
        for item in module.encoder.modules()
        if isinstance(item, DepthwiseSeparableConv)
    ]
    assert len(blocks) == 3
    assert hasattr(module.encoder, "global_gate_head")
    assert hasattr(module.encoder, "spatial_gate_head")


def test_vgup_shapes_gate_ranges_and_batch_two() -> None:
    module = VGUPPreprocessor()
    image = torch.rand(2, 3, 80, 96)
    output, debug = module(image, return_debug=True)
    assert output.shape == image.shape
    assert debug["bpw_params"].shape == (2, 12)
    assert debug["kbl_params"].shape == (2, 486)
    assert debug["global_gate"].shape == (2, 1, 1, 1)
    assert debug["spatial_gate"].shape == (2, 1, 80, 96)
    assert debug["spatial_gate_lowres"].shape[0:2] == (2, 1)
    for gate in (debug["global_gate"], debug["spatial_gate"]):
        assert gate.amin() >= 0
        assert gate.amax() <= 1
    assert torch.isfinite(output).all()


def test_vgup_residual_gate_boundaries() -> None:
    original = torch.rand(2, 3, 7, 7)
    enhanced = torch.rand(2, 3, 7, 7)
    zero_global = torch.zeros(2, 1, 1, 1)
    one_global = torch.ones(2, 1, 1, 1)
    assert torch.equal(
        VGUPPreprocessor.apply_bpw_gate(original, enhanced, zero_global),
        original,
    )
    assert torch.equal(
        VGUPPreprocessor.apply_bpw_gate(original, enhanced, one_global),
        enhanced,
    )

    zero_spatial = torch.zeros(2, 1, 7, 7)
    one_spatial = torch.ones(2, 1, 7, 7)
    assert torch.equal(
        VGUPPreprocessor.apply_kbl_gate(original, enhanced, zero_spatial),
        original,
    )
    assert torch.equal(
        VGUPPreprocessor.apply_kbl_gate(original, enhanced, one_spatial),
        enhanced,
    )


def test_vgup_all_branches_receive_gradients() -> None:
    module = VGUPPreprocessor()
    image = torch.rand(1, 3, 64, 64, requires_grad=True)
    output = module(image)
    output.square().mean().backward()
    expected = (
        module.encoder.filter_head.weight,
        module.encoder.global_gate_head.weight,
        module.encoder.spatial_gate_head.weight,
    )
    assert image.grad is not None and torch.isfinite(image.grad).all()
    for parameter in expected:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
