"""Large selective kernel operator adapted from the official LSKNet code.

Paper:
    Large Selective Kernel Network for Remote Sensing Object Detection,
    ICCV 2023.
Official source:
    https://github.com/zcablii/LSKNet
Source commit:
    386cbefc71d402e7a9375495bbe34d5c2aec0e37
License:
    CC BY-NC 4.0

The official LSK spatial-selection core is preserved. BatchNorm and SiLU are
applied after the core only to retain the output convention of the YOLO
Bottleneck.cv2 slot that this experiment replaces.
"""

from __future__ import annotations

import torch
from torch import nn


class LargeSelectiveKernelConv2d(nn.Module):
    """Official LSKblock spatial-selection core."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError(f"LSK requires a positive even channel count, got {channels}.")
        reduced_channels = channels // 2
        self.channels = channels
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
        )
        self.conv_spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=channels,
            dilation=3,
        )
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.conv_squeeze = nn.Conv2d(2, 2, kernel_size=7, padding=3)
        self.conv = nn.Conv2d(reduced_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select between local and larger-context depthwise responses."""

        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW with {self.channels} channels, got {tuple(x.shape)}."
            )
        attention_1 = self.conv0(x)
        attention_2 = self.conv_spatial(attention_1)

        attention_1 = self.conv1(attention_1)
        attention_2 = self.conv2(attention_2)
        attention = torch.cat((attention_1, attention_2), dim=1)
        average_attention = torch.mean(attention, dim=1, keepdim=True)
        maximum_attention = torch.max(attention, dim=1, keepdim=True).values
        aggregate = torch.cat((average_attention, maximum_attention), dim=1)
        selection = self.conv_squeeze(aggregate).sigmoid()
        attention = (
            attention_1 * selection[:, 0:1]
            + attention_2 * selection[:, 1:2]
        )
        return x * self.conv(attention)


class LargeSelectiveKernelConvBNAct(nn.Module):
    """YOLO cv2 adaptation: official LSK core followed by BN and SiLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.lsk = LargeSelectiveKernelConv2d(channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LSK spatial selection and the YOLO output convention."""

        return self.act(self.bn(self.lsk(x)))
