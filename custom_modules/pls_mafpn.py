"""SAF and AAF neck fusion from MAFPN adapted to the PLS P2/P3/P4 pyramid.

Paper and official source: https://github.com/yang-0201/MAF-YOLO
(commit e24674cd). RepHELAN, GHSK, backbone, and head modifications are excluded.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


def _resize_to(x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == target_size:
        return x
    height, width = x.shape[-2:]
    target_height, target_width = target_size
    if height >= target_height and width >= target_width:
        return F.adaptive_avg_pool2d(x, output_size=target_size)
    return F.interpolate(x, size=target_size, mode="nearest")


class _MAFPNFusion(nn.Module):
    """Shared target-shape alignment and concat refinement for SAF and AAF."""

    def __init__(
        self,
        input_channels: Sequence[int],
        target_index: int,
        c2: int,
        branch_channels: Sequence[int],
    ) -> None:
        super().__init__()
        channels = tuple(input_channels)
        controlled_channels = tuple(branch_channels)
        if len(channels) < 2 or len(channels) != len(controlled_channels):
            raise ValueError("MAFPN fusion requires matching input and controlled-channel lists.")
        if any(type(channel) is not int or channel <= 0 for channel in channels):
            raise TypeError(f"Feature channels must be positive Python integers, got {channels!r}.")
        if any(type(channel) is not int or channel <= 0 for channel in controlled_channels):
            raise TypeError(
                f"Controlled channels must be positive Python integers, got {controlled_channels!r}."
            )
        if type(target_index) is not int or not 0 <= target_index < len(channels):
            raise ValueError(f"Invalid target_index {target_index!r} for {len(channels)} inputs.")
        if type(c2) is not int or c2 <= 0:
            raise TypeError(f"Output channels must be a positive Python integer, got {c2!r}.")

        self.input_channels = channels
        self.target_index = target_index
        self.output_channels = c2
        self.controlled_channels = controlled_channels
        self.projections = nn.ModuleList(
            Conv(c1, controlled, k=1, s=1)
            for c1, controlled in zip(channels, controlled_channels)
        )
        self.merge = Conv(sum(controlled_channels), c2, k=1, s=1)
        self.refine = Conv(c2, c2, k=3, s=1, g=c2)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.input_channels):
            raise ValueError(f"MAFPN fusion requires {len(self.input_channels)} input feature maps.")
        target_size = tuple(features[self.target_index].shape[-2:])
        aligned = [
            _resize_to(projection(feature), target_size)
            for projection, feature in zip(self.projections, features)
        ]
        return self.refine(self.merge(torch.cat(aligned, dim=1)))


class MAFPNSAF(_MAFPNFusion):
    """Superficial Assisted Fusion with reduced high-resolution shallow branches."""

    def __init__(self, input_channels: Sequence[int], target_index: int, c2: int) -> None:
        if type(c2) is not int or c2 <= 0:
            raise TypeError(f"Output channels must be a positive Python integer, got {c2!r}.")
        channels = tuple(input_channels)
        shallow_channels = max(16, c2 // 2)
        controlled_channels = tuple(
            shallow_channels if index < target_index else c2
            for index in range(len(channels))
        )
        super().__init__(channels, target_index, c2, controlled_channels)


class MAFPNAAF(_MAFPNFusion):
    """Advanced Assisted Fusion with equal-width dense auxiliary branches."""

    def __init__(self, input_channels: Sequence[int], target_index: int, c2: int) -> None:
        if type(c2) is not int or c2 <= 0:
            raise TypeError(f"Output channels must be a positive Python integer, got {c2!r}.")
        channels = tuple(input_channels)
        super().__init__(channels, target_index, c2, (c2,) * len(channels))
