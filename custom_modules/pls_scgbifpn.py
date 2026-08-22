"""PLS adaptation of the SCGBiFPN fusion topology from ESL-YOLO.

Only spatial-context guidance and cross-level skip fusion are reproduced.
EFEM, LAPM, weighted BiFPN fusion, and detection-head changes are excluded.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


def _resize_to(x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Align to the actual reference shape without assuming an even input size."""

    if x.shape[-2:] == target_size:
        return x
    height, width = x.shape[-2:]
    target_height, target_width = target_size
    if height >= target_height and width >= target_width:
        return F.adaptive_avg_pool2d(x, output_size=target_size)
    return F.interpolate(x, size=target_size, mode="nearest")


class SCGBiFPNFusion(nn.Module):
    """Concatenate aligned branches, including direct P2 spatial-context skips."""

    def __init__(
        self,
        input_channels: Sequence[int],
        target_index: int,
        c2: int,
    ) -> None:
        super().__init__()
        channels = tuple(input_channels)
        if len(channels) < 2:
            raise ValueError("SCGBiFPNFusion requires at least two input feature maps.")
        if any(type(channel) is not int or channel <= 0 for channel in channels):
            raise TypeError(f"Feature channels must be positive Python integers, got {channels!r}.")
        if type(target_index) is not int or not 0 <= target_index < len(channels):
            raise ValueError(f"Invalid target_index {target_index!r} for {len(channels)} inputs.")
        if type(c2) is not int or c2 <= 0:
            raise TypeError(f"Output channels must be a positive Python integer, got {c2!r}.")

        self.input_channels = channels
        self.target_index = target_index
        self.output_channels = c2
        branch_channels = max(16, c2 // len(channels))
        self.branch_channels = branch_channels
        self.projections = nn.ModuleList(
            Conv(channel, branch_channels, k=1, s=1) for channel in channels
        )
        self.merge = Conv(branch_channels * len(channels), c2, k=1, s=1)
        self.spatial_refine = Conv(c2, c2, k=3, s=1, g=c2)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != len(self.input_channels):
            raise ValueError(
                f"SCGBiFPNFusion requires {len(self.input_channels)} input feature maps."
            )
        target_size = tuple(features[self.target_index].shape[-2:])
        aligned = [
            _resize_to(projection(feature), target_size)
            for projection, feature in zip(self.projections, features)
        ]
        return self.spatial_refine(self.merge(torch.cat(aligned, dim=1)))
