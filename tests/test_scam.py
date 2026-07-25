"""Unit tests for the official-flow FFCA-YOLO SCAM adapter."""

from __future__ import annotations

import torch
from torch import nn

from custom_modules.scam import ConvWithoutBN, SCAM


def test_scam_shape_and_backward() -> None:
    module = SCAM(32)
    image = torch.randn(2, 32, 9, 13, requires_grad=True)
    output = module(image)
    assert output.shape == image.shape
    output.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_scam_matches_official_softmax_and_matmul_flow() -> None:
    torch.manual_seed(0)
    module = SCAM(16).eval()
    image = torch.randn(2, 16, 7, 5)
    with torch.inference_mode():
        actual = module(image)
        batch, channels, height, width = image.shape
        avg = (
            module.avg_pool(image)
            .softmax(dim=1)
            .view(batch, 1, 1, channels)
        )
        maximum = (
            module.max_pool(image)
            .softmax(dim=1)
            .view(batch, 1, 1, channels)
        )
        k = (
            module.k(image)
            .view(batch, 1, -1, 1)
            .softmax(dim=2)
        )
        v = module.v(image).view(batch, 1, channels, -1)
        channel_context = torch.matmul(v, k).view(
            batch,
            channels,
            1,
            1,
        )
        spatial_context = torch.cat(
            (
                torch.matmul(avg, v).view(
                    batch,
                    1,
                    height,
                    width,
                ),
                torch.matmul(maximum, v).view(
                    batch,
                    1,
                    height,
                    width,
                ),
            ),
            dim=1,
        )
        expected = image + module.m(
            channel_context
        ) * module.m2(spatial_context).sigmoid()
    assert torch.equal(actual, expected)


def test_scam_m_branch_has_no_batch_norm_or_channel_reduction() -> None:
    module = SCAM(24, reduction=8)
    assert isinstance(module.m, ConvWithoutBN)
    assert not any(
        isinstance(item, nn.BatchNorm2d)
        for item in module.m.modules()
    )
    assert module.inter_channels == 24
    assert module.v.conv.out_channels == 24


def test_scam_residual_connection() -> None:
    module = SCAM(16).eval()
    nn.init.zeros_(module.m.conv.weight)
    image = torch.randn(1, 16, 8, 8)
    with torch.inference_mode():
        assert torch.equal(module(image), image)


def test_three_scam_instances_do_not_share_parameters() -> None:
    modules = [SCAM(16) for _ in range(3)]
    parameter_ids = [
        {id(parameter) for parameter in module.parameters()}
        for module in modules
    ]
    assert parameter_ids[0].isdisjoint(parameter_ids[1])
    assert parameter_ids[0].isdisjoint(parameter_ids[2])
    assert parameter_ids[1].isdisjoint(parameter_ids[2])
