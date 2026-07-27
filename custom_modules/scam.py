"""SCAM adapted from the official FFCA-YOLO implementation.

Paper: FFCA-YOLO for Small Object Detection in Remote Sensing Images
Journal: IEEE Transactions on Geoscience and Remote Sensing, 2024
Official repository: https://github.com/yemu1138178251/FFCA-YOLO
Reference commit: 874a00da12266b4ee1abc3b6494c193972488956

The official k/v projections, GAP/GMP channel softmax, spatial softmax,
matrix multiplications, multiplicative gate, no-BN ``m`` projection, and
``x + y`` residual are preserved. No channel reduction is introduced.
"""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import Conv


class ConvWithoutBN(nn.Module):
    """Official FFCA-YOLO no-BN convolution used by SCAM's m branch."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SCAM(nn.Module):
    """Faithful shape-preserving SCAM adapter for YOLO feature maps."""

    def __init__(
        self,
        in_channels: int,
        reduction: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.inter_channels = in_channels
        # The official constructor accepts reduction but does not use it for
        # channel compression. Retaining it preserves the public interface.
        self.reduction = reduction

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = ConvWithoutBN(self.inter_channels, in_channels, 1, 1)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def compute_context_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return the original FFCA-YOLO context residual before ``x + y``."""

        if x.ndim != 4:
            raise ValueError(
                f"SCAM expects a BCHW tensor, got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SCAM expected {self.in_channels} channels, got {x.shape[1]}."
            )

        batch, channels, height, width = x.shape

        avg = (
            self.avg_pool(x)
            .softmax(dim=1)
            .view(batch, 1, 1, channels)
        )
        maximum = (
            self.max_pool(x)
            .softmax(dim=1)
            .view(batch, 1, 1, channels)
        )

        k = (
            self.k(x)
            .view(batch, 1, -1, 1)
            .softmax(dim=2)
        )
        v = self.v(x).view(batch, 1, channels, -1)

        channel_context = torch.matmul(v, k).view(
            batch,
            channels,
            1,
            1,
        )
        spatial_avg = torch.matmul(avg, v).view(
            batch,
            1,
            height,
            width,
        )
        spatial_max = torch.matmul(maximum, v).view(
            batch,
            1,
            height,
            width,
        )
        spatial_context = torch.cat(
            (spatial_avg, spatial_max),
            dim=1,
        )

        y = self.m(channel_context) * self.m2(spatial_context).sigmoid()
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.compute_context_residual(x)
