"""ECF-YOLO CGFM adaptation for YOLO11.

Paper-faithful reimplementation for YOLO11. No verifiable complete official
author implementation was available when this adapter was prepared.
"""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import Conv


def _unpack_inputs(inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        raise ValueError("CGFM/AlignConcat requires [deep_upsampled, shallow_lateral].")
    deep, shallow = inputs
    if deep.shape[-2:] != shallow.shape[-2:]:
        raise ValueError(
            "CGFM/AlignConcat inputs must have identical spatial sizes; "
            f"got deep={tuple(deep.shape[-2:])}, shallow={tuple(shallow.shape[-2:])}."
        )
    return deep, shallow


class CGFM(nn.Module):
    """Channel-guided fusion with dual pooling and cross-residual enhancement."""

    def __init__(self, c_deep: int, c_shallow: int, reduction: int = 16) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        self.c_shallow = c_shallow
        self.align = Conv(c_deep, c_shallow, k=1, s=1)
        hidden = max(1, (2 * c_shallow) // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_branch = nn.Sequential(
            nn.Conv2d(2 * c_shallow, hidden, 1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, c_shallow, 1, bias=True),
        )
        self.max_branch = nn.Sequential(
            nn.Conv2d(2 * c_shallow, hidden, 1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, c_shallow, 1, bias=True),
        )
        self.gate = nn.Sigmoid()

    def channel_weights(self, fused: torch.Tensor) -> torch.Tensor:
        """Return W with shape B x C2 x 1 x 1."""
        return self.gate(
            self.avg_branch(self.avg_pool(fused))
            + self.max_branch(self.max_pool(fused))
        )

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        x1, x2 = _unpack_inputs(inputs)
        x3 = self.align(x1)
        x4 = torch.cat((x3, x2), dim=1)
        weights = self.channel_weights(x4)
        x6 = x3 * weights
        x7 = x2 * weights
        y1 = x6 + x2
        y2 = x7 + x3
        return torch.cat((y1, y2), dim=1)


class AlignConcat(nn.Module):
    """Diagnostic control: channel alignment followed by plain concatenation."""

    def __init__(self, c_deep: int, c_shallow: int) -> None:
        super().__init__()
        self.align = Conv(c_deep, c_shallow, k=1, s=1)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        x1, x2 = _unpack_inputs(inputs)
        return torch.cat((self.align(x1), x2), dim=1)
