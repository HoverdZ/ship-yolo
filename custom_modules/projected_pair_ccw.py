"""Lite-HRNet conditional channel weighting adapted for DPLS pair inputs.

The weighting path follows the official Lite-HRNet implementation while the
``ProjectedPairCCW`` wrapper adds the channel projection required after a YOLO
Concat. Only the refined target branch is returned; the lower-resolution
context branch conditions its weights.

Reference:
https://github.com/HRNet/Lite-HRNet/blob/7b9049d264fa40402a27d1f175deff3b46a6b91b/models/backbones/litehrnet.py
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels between groups without changing tensor shape."""

    if x.ndim != 4:
        raise ValueError(f"channel_shuffle expects a 4D tensor, got {x.ndim}D.")
    batch, channels, height, width = x.shape
    if groups < 1 or channels % groups:
        raise ValueError(f"{channels} channels cannot be shuffled into {groups} groups.")
    channels_per_group = channels // groups
    return (
        x.reshape(batch, groups, channels_per_group, height, width)
        .transpose(1, 2)
        .reshape(batch, channels, height, width)
    )


class SpatialWeighting(nn.Module):
    """Lite-HRNet spatial weighting for one active channel branch."""

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        if channels < 1 or ratio < 1:
            raise ValueError("channels and ratio must be positive integers.")
        hidden_channels = max(channels // ratio, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, hidden_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.global_avgpool(x)
        weights = self.relu(self.conv1(weights))
        weights = self.sigmoid(self.conv2(weights))
        return x * weights


class CrossResolutionWeighting(nn.Module):
    """Generate per-channel weights jointly from multiple resolutions."""

    def __init__(self, channels: Sequence[int], ratio: int = 16) -> None:
        super().__init__()
        self.channels = tuple(int(channel) for channel in channels)
        if len(self.channels) < 2 or any(channel < 1 for channel in self.channels):
            raise ValueError("CrossResolutionWeighting requires at least two positive channel widths.")
        if ratio < 1:
            raise ValueError("ratio must be a positive integer.")

        total_channels = sum(self.channels)
        hidden_channels = max(total_channels // ratio, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(total_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, total_channels, 1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not isinstance(x, (list, tuple)) or len(x) != len(self.channels):
            raise ValueError(f"Expected {len(self.channels)} resolution branches.")
        for branch, expected_channels in zip(x, self.channels):
            if branch.ndim != 4 or branch.shape[1] != expected_channels:
                raise ValueError(
                    "CrossResolutionWeighting received a branch with an unexpected shape: "
                    f"expected {expected_channels} channels, got {tuple(branch.shape)}."
                )

        lowest_size = x[-1].shape[-2:]
        pooled = [F.adaptive_avg_pool2d(branch, lowest_size) for branch in x[:-1]]
        pooled.append(x[-1])
        weights = self.conv2(self.conv1(torch.cat(pooled, dim=1)))
        weights = torch.split(weights, self.channels, dim=1)
        return [
            branch * F.interpolate(weight, size=branch.shape[-2:], mode="nearest")
            for branch, weight in zip(x, weights)
        ]


class ConditionalChannelWeighting(nn.Module):
    """Lite-HRNet CCW block with a bypass half and an active half per branch."""

    def __init__(self, in_channels: Sequence[int], reduce_ratio: int = 8) -> None:
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        if len(self.in_channels) < 2 or any(
            channel < 2 or channel % 2 for channel in self.in_channels
        ):
            raise ValueError("Every CCW branch must have a positive, even channel width.")

        active_channels = [channel // 2 for channel in self.in_channels]
        self.cross_resolution_weighting = CrossResolutionWeighting(
            active_channels,
            ratio=reduce_ratio,
        )
        self.depthwise_convs = nn.ModuleList(
            Conv(channel, channel, k=3, s=1, g=channel, act=False)
            for channel in active_channels
        )
        self.spatial_weighting = nn.ModuleList(
            SpatialWeighting(channel, ratio=4) for channel in active_channels
        )

    def forward(self, x: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not isinstance(x, (list, tuple)) or len(x) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} CCW branches.")

        split_branches = [branch.chunk(2, dim=1) for branch in x]
        bypass = [parts[0] for parts in split_branches]
        active = [parts[1] for parts in split_branches]
        active = self.cross_resolution_weighting(active)
        active = [conv(branch) for branch, conv in zip(active, self.depthwise_convs)]
        active = [weight(branch) for branch, weight in zip(active, self.spatial_weighting)]
        return [
            channel_shuffle(torch.cat((skip, branch), dim=1), groups=2)
            for skip, branch in zip(bypass, active)
        ]


class ProjectedPairCCW(nn.Module):
    """Project a fused target and refine it using a lower-resolution context.

    Args:
        in_channels: ``(target_channels, context_channels)`` supplied by
            ``parse_model`` from the ordered YAML ``from`` list.
        out_channels: Output width of the target branch after projection.
        reduce_ratio: Bottleneck ratio for cross-resolution weighting.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        reduce_ratio: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) != 2:
            raise ValueError("ProjectedPairCCW requires [target, context] input channels.")
        target_channels, context_channels = (int(channel) for channel in in_channels)
        out_channels = int(out_channels)
        if target_channels < 1 or context_channels < 1 or out_channels < 1:
            raise ValueError("All ProjectedPairCCW channel widths must be positive.")
        if out_channels % 2 or context_channels % 2:
            raise ValueError("Projected target and context channel widths must be even for CCW.")

        self.target_channels = target_channels
        self.context_channels = context_channels
        self.out_channels = out_channels
        self.project = (
            Conv(target_channels, out_channels, k=1, s=1)
            if target_channels != out_channels
            else nn.Identity()
        )
        self.ccw = ConditionalChannelWeighting(
            [out_channels, context_channels],
            reduce_ratio=reduce_ratio,
        )

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("ProjectedPairCCW expects the ordered pair [target, context].")
        target, context = x
        if target.ndim != 4 or context.ndim != 4:
            raise ValueError("ProjectedPairCCW inputs must be 4D NCHW tensors.")
        if target.shape[0] != context.shape[0]:
            raise ValueError("Target and context batch sizes must match.")
        if target.shape[1] != self.target_channels or context.shape[1] != self.context_channels:
            raise ValueError(
                "ProjectedPairCCW input channels do not match parse_model inference: "
                f"got ({target.shape[1]}, {context.shape[1]}), expected "
                f"({self.target_channels}, {self.context_channels})."
            )
        if any(
            target_size < context_size
            for target_size, context_size in zip(target.shape[-2:], context.shape[-2:])
        ):
            raise ValueError("Context must not have a higher spatial resolution than target.")

        target = self.project(target)
        return self.ccw([target, context])[0]
