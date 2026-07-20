"""Spatial and Channel Reconstruction Convolution for YOLO11 experiments.

This is a repository-owned reproduction of "SCConv: Spatial and Channel
Reconstruction Convolution for Feature Redundancy" (CVPR 2023). The structure
follows the paper and the public, non-official ``cheng-haha/ScConv`` reference,
with validation, numerically stable normalization, and device-agnostic tensor
operations added for this project. It is not official code from the authors.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


class GroupBatchnorm2d(nn.Module):
    """Group-wise normalization used by the non-native SRU option.

    The name is retained for compatibility with common SCConv reproductions.
    Its operation is group normalization over channel/spatial elements, not
    running-statistics BatchNorm.
    """

    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-5) -> None:
        super().__init__()
        _require_positive_int("c_num", c_num)
        _require_positive_int("group_num", group_num)
        if c_num % group_num != 0:
            raise ValueError(
                f"GroupBatchnorm2d requires c_num divisible by group_num, got "
                f"c_num={c_num}, group_num={group_num}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.c_num = c_num
        self.group_num = group_num
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c_num, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a BCHW tensor without creating device-specific tensors."""

        if x.ndim != 4:
            raise ValueError(f"GroupBatchnorm2d expects BCHW input, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.c_num:
            raise ValueError(
                f"GroupBatchnorm2d was built for {self.c_num} channels, got {x.shape[1]}."
            )

        batch, channels, height, width = x.shape
        grouped = x.reshape(batch, self.group_num, -1)
        mean = grouped.mean(dim=2, keepdim=True)
        variance = grouped.var(dim=2, keepdim=True, unbiased=False)
        normalized = (grouped - mean) * torch.rsqrt(variance + self.eps)
        normalized = normalized.reshape(batch, channels, height, width)
        return normalized * self.weight + self.bias


class SRU(nn.Module):
    """Spatial Reconstruction Unit using separate-and-reconstruct."""

    def __init__(
        self,
        out_channels: int,
        group_num: int = 4,
        gate_threshold: float = 0.5,
        torch_gn: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        _require_positive_int("out_channels", out_channels)
        _require_positive_int("group_num", group_num)
        if out_channels % 2 != 0:
            raise ValueError(
                f"SRU cross-reconstruction requires an even channel count, got {out_channels}."
            )
        if out_channels % group_num != 0:
            raise ValueError(
                f"SRU requires out_channels divisible by group_num, got "
                f"out_channels={out_channels}, group_num={group_num}."
            )
        if not 0.0 <= gate_threshold <= 1.0:
            raise ValueError(f"gate_threshold must be in [0, 1], got {gate_threshold}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.out_channels = out_channels
        self.group_num = group_num
        self.gate_threshold = float(gate_threshold)
        self.eps = float(eps)
        self.norm: nn.Module
        if torch_gn:
            self.norm = nn.GroupNorm(group_num, out_channels, eps=eps, affine=True)
        else:
            self.norm = GroupBatchnorm2d(out_channels, group_num, eps)

    def _normalized_gamma(self) -> torch.Tensor:
        gamma = self.norm.weight.reshape(-1)
        denominator = gamma.sum()
        safe_denominator = torch.where(
            denominator.abs() >= self.eps,
            denominator,
            torch.full_like(denominator, self.eps),
        )
        return gamma / safe_denominator

    def reconstruct(self, informative: torch.Tensor, redundant: torch.Tensor) -> torch.Tensor:
        """Cross-add channel halves and concatenate them."""

        informative_first, informative_second = informative.chunk(2, dim=1)
        redundant_first, redundant_second = redundant.chunk(2, dim=1)
        return torch.cat(
            (
                informative_first + redundant_second,
                informative_second + redundant_first,
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial separation and cross reconstruction."""

        if x.ndim != 4 or x.shape[1] != self.out_channels:
            raise ValueError(
                f"SRU expects BCHW input with {self.out_channels} channels, got {tuple(x.shape)}."
            )

        normalized = self.norm(x)
        gamma = self._normalized_gamma().view(1, -1, 1, 1)
        importance = torch.sigmoid(normalized * gamma)
        informative_weights = torch.where(
            importance >= self.gate_threshold,
            torch.ones_like(importance),
            importance,
        )
        redundant_weights = torch.where(
            importance >= self.gate_threshold,
            torch.zeros_like(importance),
            importance,
        )
        return self.reconstruct(informative_weights * x, redundant_weights * x)


class CRU(nn.Module):
    """Channel Reconstruction Unit using split-transform-and-fuse."""

    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        _require_positive_int("channels", channels)
        out_channels = channels if out_channels is None else out_channels
        _require_positive_int("out_channels", out_channels)
        _require_positive_int("squeeze_ratio", squeeze_ratio)
        _require_positive_int("group_size", group_size)
        _require_positive_int("group_kernel_size", group_kernel_size)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")
        if group_kernel_size % 2 == 0:
            raise ValueError(
                f"group_kernel_size must be odd to preserve spatial size, got {group_kernel_size}."
            )

        upper_channels = int(alpha * channels)
        lower_channels = channels - upper_channels
        if upper_channels <= 0 or lower_channels <= 0:
            raise ValueError(
                f"alpha={alpha} creates an empty CRU branch for channels={channels}."
            )
        upper_squeezed = upper_channels // squeeze_ratio
        lower_squeezed = lower_channels // squeeze_ratio
        if upper_squeezed <= 0 or lower_squeezed <= 0:
            raise ValueError(
                f"squeeze_ratio={squeeze_ratio} creates zero channels: "
                f"upper={upper_squeezed}, lower={lower_squeezed}."
            )
        if upper_squeezed % group_size != 0 or out_channels % group_size != 0:
            raise ValueError(
                "CRU grouped convolution requires both squeezed input and output channels "
                f"divisible by group_size; got upper_squeezed={upper_squeezed}, "
                f"out_channels={out_channels}, group_size={group_size}."
            )
        lower_projected = out_channels - lower_squeezed
        if lower_projected <= 0:
            raise ValueError(
                f"CRU lower projection would have {lower_projected} output channels."
            )

        self.channels = channels
        self.out_channels = out_channels
        self.alpha = float(alpha)
        self.squeeze_ratio = squeeze_ratio
        self.group_size = group_size
        self.upper_channels = upper_channels
        self.lower_channels = lower_channels
        self.upper_squeezed = upper_squeezed
        self.lower_squeezed = lower_squeezed

        self.squeeze_upper = nn.Conv2d(upper_channels, upper_squeezed, 1, bias=False)
        self.squeeze_lower = nn.Conv2d(lower_channels, lower_squeezed, 1, bias=False)
        self.gwc = nn.Conv2d(
            upper_squeezed,
            out_channels,
            group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=group_size,
            bias=False,
        )
        self.pwc_upper = nn.Conv2d(upper_squeezed, out_channels, 1, bias=False)
        self.pwc_lower = nn.Conv2d(lower_squeezed, lower_projected, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CRU channel split, lightweight transforms, and soft fusion."""

        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"CRU expects BCHW input with {self.channels} channels, got {tuple(x.shape)}."
            )

        upper, lower = torch.split(
            x, (self.upper_channels, self.lower_channels), dim=1
        )
        upper = self.squeeze_upper(upper)
        lower = self.squeeze_lower(lower)
        upper_output = self.gwc(upper) + self.pwc_upper(upper)
        lower_output = torch.cat((self.pwc_lower(lower), lower), dim=1)
        combined = torch.cat((upper_output, lower_output), dim=1)
        weighted = F.softmax(self.pool(combined), dim=1) * combined
        first, second = torch.split(weighted, self.out_channels, dim=1)
        return first + second


class ScConv(nn.Module):
    """Sequential SRU and CRU replacement for a 3x3 convolution.

    ``out_channels`` defaults to ``channels``. The optional C1-to-C2 form follows
    the paper's general CRU equations and is needed to preserve YOLO bottleneck
    expansion without adding an external projection.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        group_num: int = 4,
        gate_threshold: float = 0.5,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
        torch_gn: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = channels if out_channels is None else out_channels
        self.sru = SRU(
            channels,
            group_num=group_num,
            gate_threshold=gate_threshold,
            torch_gn=torch_gn,
            eps=eps,
        )
        self.cru = CRU(
            channels,
            out_channels=self.out_channels,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio,
            group_size=group_size,
            group_kernel_size=group_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a feature map with unchanged spatial size."""

        return self.cru(self.sru(x))
