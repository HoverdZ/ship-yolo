"""HHSPP with a context-gated local-detail compensation branch."""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

from custom_modules.hhspp import HHSPP


class HHSPPLocalDetail(HHSPP):
    """Preserve HHSPP context features while progressively restoring local detail."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__(c1, c2)
        detail_channels = max(16, c1 // 4)
        if not isinstance(detail_channels, int):
            raise TypeError(
                f"HHSPPLocalDetail requires integer channels, got {detail_channels!r}."
            )

        self.detail_channels = detail_channels
        self.detail_reduce = Conv(c1, detail_channels, k=1, s=1)
        self.detail_dw = Conv(
            detail_channels,
            detail_channels,
            k=3,
            s=1,
            g=detail_channels,
        )
        self.detail_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.detail_proj = Conv(detail_channels, c2, k=1, s=1)
        self.detail_gate = nn.Conv2d(
            c2,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.detail_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = super().forward(x)
        z = self.detail_reduce(x)
        local = self.detail_dw(z)
        high = z - self.detail_avg(z)
        detail = self.detail_proj(local + high)
        gate = torch.sigmoid(self.detail_gate(context))
        return context + self.detail_scale * gate * detail
