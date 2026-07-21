"""Inception depthwise convolution building blocks.

Core InceptionDWConv2d design adapted from:
InceptionNeXt: When Inception Meets ConvNeXt, CVPR 2024
Official implementation: sail-sg/inceptionnext
License: Apache-2.0

The core below intentionally preserves the official four-way channel split and
default hyperparameters. Repository-specific channel adaptation, when needed,
is kept outside this class so the core remains directly testable against the
official implementation.
"""

from __future__ import annotations

import torch
from torch import nn


class InceptionDWConv2d(nn.Module):
    """Official InceptionNeXt InceptionDWConv2d core."""

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if square_kernel_size <= 0 or square_kernel_size % 2 == 0:
            raise ValueError("square_kernel_size must be a positive odd integer.")
        if band_kernel_size <= 0 or band_kernel_size % 2 == 0:
            raise ValueError("band_kernel_size must be a positive odd integer.")
        if not 0.0 < branch_ratio <= 1.0 / 3.0:
            raise ValueError("branch_ratio must be in (0, 1/3].")

        branch_channels = int(in_channels * branch_ratio)
        identity_channels = in_channels - 3 * branch_channels
        if branch_channels < 1 or identity_channels < 0:
            raise ValueError(
                "InceptionDWConv2d requires at least one channel in every convolution branch; "
                f"got in_channels={in_channels}, branch_ratio={branch_ratio}."
            )

        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.dwconv_hw = nn.Conv2d(
            branch_channels,
            branch_channels,
            square_kernel_size,
            padding=square_kernel_size // 2,
            groups=branch_channels,
        )
        self.dwconv_w = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=branch_channels,
        )
        self.dwconv_h = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=branch_channels,
        )
        self.split_indexes = (identity_channels, branch_channels, branch_channels, branch_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split, transform three branches, and concatenate in official order."""

        if x.ndim != 4:
            raise ValueError(f"Expected a BCHW tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[1]}.")
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (
                x_id,
                self.dwconv_hw(x_hw),
                self.dwconv_w(x_w),
                self.dwconv_h(x_h),
            ),
            dim=1,
        )


class InceptionDWConvBNAct(nn.Module):
    """Apply the exact InceptionDWConv2d -> BatchNorm2d -> SiLU sequence."""

    def __init__(
        self,
        channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        self.inception = InceptionDWConv2d(
            channels,
            square_kernel_size=square_kernel_size,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped operator in the declared order."""

        return self.act(self.bn(self.inception(x)))
