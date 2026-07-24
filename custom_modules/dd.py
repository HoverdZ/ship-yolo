"""TD-Net DD adaptation for YOLO11.

Paper-faithful reimplementation for YOLO11. No verifiable complete official
author implementation was available when this adapter was prepared.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules import Conv


class Extract(nn.Module):
    """Shared DPL extraction function E: 1x1 Conv-BN then 1x1 Conv-SiLU."""

    def __init__(self, channels: int, r: int = 4) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}.")
        hidden = max(1, channels // r)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.bn(self.conv1(x))))


class ECALayer(nn.Module):
    """Efficient channel attention used by the DPL fusion path."""

    def __init__(self, channels: int, gamma: int = 2, bias: int = 1) -> None:
        super().__init__()
        kernel_size = int(abs((math.log2(max(channels, 1)) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.gate(self.conv(weights).transpose(-1, -2).unsqueeze(-1))
        return x * weights.expand_as(x)


class DPL(nn.Module):
    """Defect perception and information-complement branch."""

    def __init__(self, c1: int, c2: int, r: int = 4) -> None:
        super().__init__()
        self.extract = Extract(c1, r=r)
        self.fuse1 = nn.Conv2d(4 * c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.eca = ECALayer(c2)
        self.fuse2 = nn.Conv2d(c2, c2, 1, bias=True)
        self.act = nn.SiLU()

    @staticmethod
    def _pad_even(x: torch.Tensor) -> torch.Tensor:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        return F.pad(x, (0, pad_w, 0, pad_h)) if pad_h or pad_w else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_even(x)
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]

        e0 = self.extract(x0)
        e1 = self.extract(x1 - e0)
        e2 = self.extract(x2 - e0)
        e3 = self.extract(x3 - e0)
        fused = torch.cat((e0, e1, e2, e3), dim=1)
        return self.act(self.fuse2(self.eca(self.bn(self.fuse1(fused)))))


class DD(nn.Module):
    """Defect Downsampling: stride-2 Conv branch plus additive DPL branch."""

    def __init__(self, c1: int, c2: int, r: int = 4) -> None:
        super().__init__()
        self.conv = Conv(c1, c2, k=3, s=2)
        self.dpl = DPL(c1, c2, r=r)

    def branch_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both branches for validation without changing forward mathematics."""
        return self.conv(x), self.dpl(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_output, dpl_output = self.branch_outputs(x)
        if conv_output.shape != dpl_output.shape:
            raise RuntimeError(
                "DD branch shape mismatch: "
                f"Conv={tuple(conv_output.shape)}, DPL={tuple(dpl_output.shape)}."
            )
        return conv_output + dpl_output
