"""Unit tests for the official-logic DySample adapter."""

from __future__ import annotations

import pytest
import torch

from custom_modules.dysample import DySample


@pytest.mark.parametrize("channels", [16, 32, 64])
def test_dysample_lp_doubles_spatial_shape(channels: int) -> None:
    module = DySample(
        channels,
        scale=2,
        style="lp",
        groups=4,
        dyscope=False,
    )
    image = torch.randn(2, channels, 7, 11)
    output = module(image)
    assert output.shape == (2, channels, 14, 22)


def test_dysample_rejects_channels_not_divisible_by_groups() -> None:
    with pytest.raises(ValueError, match="divisible by groups"):
        DySample(18, groups=4)


def test_dysample_backward_is_finite() -> None:
    module = DySample(16, groups=4)
    image = torch.randn(1, 16, 8, 8, requires_grad=True)
    module(image).square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_dysample_preserves_official_initialization() -> None:
    module = DySample(16, scale=2, style="lp", groups=4)
    assert module.offset.weight.std().item() < 0.002
    expected = torch.tensor(
        [-0.25, 0.25, -0.25, 0.25],
        dtype=module.init_pos.dtype,
    )
    assert torch.equal(module.init_pos[0, :4, 0, 0], expected)


def test_dysample_has_no_custom_cuda_dependency() -> None:
    module = DySample(16)
    output = module(torch.randn(1, 16, 4, 4))
    assert output.device.type == "cpu"
