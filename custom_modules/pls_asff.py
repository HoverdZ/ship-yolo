"""ASFF adapted to the P2/P3/P4 feature pyramid used by PLS.

Paper: Learning Spatial Fusion for Single-Shot Object Detection.
Official source: https://github.com/ruinmessi/ASFF (commit 4df6f728).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


def _checked_channels(channels: Sequence[int], expected: int | None = None) -> tuple[int, ...]:
    values = tuple(channels)
    if expected is not None and len(values) != expected:
        raise ValueError(f"Expected {expected} feature levels, got {len(values)}.")
    if not values or any(type(channel) is not int or channel <= 0 for channel in values):
        raise TypeError(f"Feature channels must be positive Python integers, got {values!r}.")
    return values


class _ASFFAlign(nn.Module):
    """Apply the resize rule used by ASFF for one source-to-target level pair."""

    def __init__(self, c1: int, c2: int, source_level: int, target_level: int) -> None:
        super().__init__()
        level_delta = target_level - source_level
        if level_delta == 2:
            self.transform = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Conv(c1, c2, k=3, s=2),
            )
        elif level_delta == 1:
            self.transform = Conv(c1, c2, k=3, s=2)
        else:
            self.transform = Conv(c1, c2, k=1, s=1)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.transform(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="nearest")
        return x


class PLSASFF(nn.Module):
    """Spatially weight all P2/P3/P4 inputs at one requested pyramid level."""

    def __init__(
        self,
        input_channels: Sequence[int],
        target_index: int,
        c2: int,
    ) -> None:
        super().__init__()
        channels = _checked_channels(input_channels, expected=3)
        if type(target_index) is not int or not 0 <= target_index < len(channels):
            raise ValueError(f"target_index must be 0, 1, or 2, got {target_index!r}.")
        if type(c2) is not int or c2 <= 0:
            raise TypeError(f"Output channels must be a positive Python integer, got {c2!r}.")

        self.input_channels = channels
        self.target_index = target_index
        self.output_channels = c2
        self.align = nn.ModuleList(
            _ASFFAlign(channel, c2, source_index, target_index)
            for source_index, channel in enumerate(channels)
        )

        compressed_channels = min(16, c2)
        self.weight_features = nn.ModuleList(
            Conv(c2, compressed_channels, k=1, s=1) for _ in channels
        )
        self.weight_levels = nn.Conv2d(
            compressed_channels * len(channels),
            len(channels),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.refine = Conv(c2, c2, k=3, s=1)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.input_channels):
            raise ValueError(f"PLSASFF requires {len(self.input_channels)} input feature maps.")
        target_size = tuple(features[self.target_index].shape[-2:])
        aligned = [
            operation(feature, target_size)
            for operation, feature in zip(self.align, features)
        ]
        weight_vectors = [
            operation(feature)
            for operation, feature in zip(self.weight_features, aligned)
        ]
        logits = self.weight_levels(torch.cat(weight_vectors, dim=1))
        weights = torch.softmax(logits.float(), dim=1).to(dtype=aligned[0].dtype)
        fused = (
            aligned[0] * weights[:, 0:1]
            + aligned[1] * weights[:, 1:2]
            + aligned[2] * weights[:, 2:3]
        )
        return self.refine(fused)
