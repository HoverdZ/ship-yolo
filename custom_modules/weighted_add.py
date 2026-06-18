"""Static learnable weighted feature fusion modules for WAFPN experiments."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules import Conv
except Exception:  # pragma: no cover - used only outside an Ultralytics source tree
    Conv = None


class _FallbackConv(nn.Module):
    """Small Conv-BN-SiLU block used when Ultralytics Conv is unavailable."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1) -> None:
        super().__init__()
        padding = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, padding, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def _conv(c1: int, c2: int, k: int = 1, s: int = 1) -> nn.Module:
    if Conv is not None:
        return Conv(c1, c2, k, s)
    return _FallbackConv(c1, c2, k, s)


class WeightedAdd2(nn.Module):
    """Fuse two same-shape feature maps with normalized learnable weights."""

    n_inputs = 2

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)):
            raise TypeError(f"WeightedAdd2 expects list/tuple inputs, got {type(xs).__name__}")
        if len(xs) != self.n_inputs:
            raise ValueError(f"WeightedAdd2 expects exactly 2 inputs, got {len(xs)}")

        x1, x2 = xs
        if x1.shape != x2.shape:
            raise ValueError(
                "WeightedAdd2 requires equal input shapes, "
                f"got {tuple(x1.shape)} and {tuple(x2.shape)}"
            )

        weights = torch.relu(self.w)
        return (weights[0] * x1 + weights[1] * x2) / (weights.sum() + self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, n_inputs={self.n_inputs}"


class AlignWeightedAdd2(nn.Module):
    """Align two feature maps by channel and size, then apply WeightedAdd2."""

    def __init__(
        self,
        c1: int,
        c2: int,
        c_out: int,
        eps: float = 1e-4,
        use_post_conv: bool = True,
    ) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c_out = c_out
        self.eps = eps
        self.use_post_conv = use_post_conv

        self.align1 = _conv(c1, c_out, 1, 1)
        self.align2 = _conv(c2, c_out, 1, 1)
        self.fuse = WeightedAdd2(eps=eps)
        self.post = _conv(c_out, c_out, 3, 1) if use_post_conv else nn.Identity()

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)):
            raise TypeError(f"AlignWeightedAdd2 expects list/tuple inputs, got {type(xs).__name__}")
        if len(xs) != 2:
            raise ValueError(f"AlignWeightedAdd2 expects exactly 2 inputs, got {len(xs)}")

        x1, x2 = xs
        y1 = self.align1(x1)
        y2 = self.align2(x2)

        if y2.shape[-2:] != y1.shape[-2:]:
            y2 = F.interpolate(y2, size=y1.shape[-2:], mode="nearest")

        return self.post(self.fuse([y1, y2]))

    def extra_repr(self) -> str:
        return (
            f"c1={self.c1}, c2={self.c2}, c_out={self.c_out}, "
            f"eps={self.eps}, use_post_conv={self.use_post_conv}"
        )
