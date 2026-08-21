"""HHSPP migrated from the official DPCSANet-2025 implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class HHSPP(nn.Module):
    """Hierarchical hybrid spatial pyramid pooling used by DPCSANet."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        hidden = c1 // 2
        if hidden <= 0:
            raise ValueError(f"HHSPP requires a positive input channel count, got {c1}.")
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * 5, c2, 1, 1)
        self.pool5_first = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool5_second = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.cv1(x)
        y1 = self.pool5_first(y0)
        y2 = self.pool5_second(y1)
        y3 = self.pool9(y1)
        y4 = self.pool13(y1)
        return self.cv2(torch.cat((y0, y1, y2, y3, y4), dim=1))
