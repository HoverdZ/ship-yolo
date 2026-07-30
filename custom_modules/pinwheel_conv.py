"""Pinwheel-shaped convolution adapted from the official AAAI 2025 code.

Paper:
    Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared
    Small Target Detection, AAAI 2025.
Official source:
    https://github.com/JN-Yang/PConv-SDloss-Data
Source commit:
    a801f043c83f73aa9af9ab2f689e59ebef928fc4
License:
    MIT

Only the PConv operator is used. The paper's SD loss is intentionally excluded
so this repository can test one structural variable at a time.
"""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import Conv


class PinwheelConv(nn.Module):
    """Official four-direction asymmetric-padding PConv operator."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"Channels must be positive, got c1={c1}, c2={c2}.")
        if c2 % 4:
            raise ValueError(f"PinwheelConv requires c2 divisible by 4, got {c2}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if stride != 1:
            raise ValueError("This controlled C3k2 experiment requires stride=1.")

        k = kernel_size
        paddings = ((k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k))
        self.pad = nn.ModuleList(nn.ZeroPad2d(padding) for padding in paddings)
        self.cw = Conv(c1, c2 // 4, (1, k), s=stride, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=stride, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate four asymmetric horizontal and vertical responses."""

        horizontal_0 = self.cw(self.pad[0](x))
        horizontal_1 = self.cw(self.pad[1](x))
        vertical_0 = self.ch(self.pad[2](x))
        vertical_1 = self.ch(self.pad[3](x))
        return self.cat(
            torch.cat(
                (horizontal_0, horizontal_1, vertical_0, vertical_1),
                dim=1,
            )
        )
