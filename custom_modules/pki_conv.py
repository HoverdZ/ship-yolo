"""Poly-kernel convolutional mixer adapted from the official PKINet code.

Paper:
    Poly Kernel Inception Network for Remote Sensing Detection, CVPR 2024.
Official source:
    https://github.com/PKINet/PKINet
Source commit:
    a33aa22d188c9946cc83fba60e3bb8ac0ec82ff7
License:
    Apache-2.0

This file extracts the paper's dense, non-dilated poly-kernel convolutional
mixer. Context Anchor Attention, ConvFFN, DropPath and other PKIBlock
components are deliberately excluded so the experiment changes only the
Bottleneck.cv2 convolutional operator.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from ultralytics.nn.modules import Conv


class PolyKernelConv2d(nn.Module):
    """Dense multi-kernel depthwise mixer from PKINet's inception bottleneck."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
    ) -> None:
        super().__init__()
        kernels = tuple(int(kernel) for kernel in kernel_sizes)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if len(kernels) < 2:
            raise ValueError("PolyKernelConv2d requires at least two kernels.")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError(f"All kernels must be positive odd integers, got {kernels}.")

        self.channels = channels
        self.kernel_sizes = kernels
        self.base_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernels[0],
            stride=1,
            padding=kernels[0] // 2,
            groups=channels,
            bias=False,
        )
        self.branch_convs = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                groups=channels,
                bias=False,
            )
            for kernel in kernels[1:]
        )
        self.project = Conv(channels, channels, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dense poly-kernel mixing without sparse dilation."""

        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW with {self.channels} channels, got {tuple(x.shape)}."
            )
        base = self.base_conv(x)
        mixed = base
        for branch in self.branch_convs:
            mixed = mixed + branch(base)
        return self.project(mixed)
