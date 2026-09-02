"""Controlled lightweight convolution operators for the DPLS neck.

The C3k2 variants preserve the Ultralytics 8.4.92 C2f/C3k2 container,
projections, concatenation, shortcut rule, and each Bottleneck ``cv1``. Only
the second spatial convolution (``Bottleneck.cv2``) changes.

Official implementation references:
- GSConv: https://github.com/AlanLi1997/slim-neck-by-gsconv/blob/master/
  gsconv-yolov8_9_10_11/ultralytics/nn/modules/conv.py
- FasterNet Partial_conv3: https://github.com/JierunChen/FasterNet/blob/
  master/models/fasternet.py
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from ultralytics.nn.modules import C2f, Conv, DWConv, GhostConv


class GSConv(nn.Module):
    """Original GSConv: primary Conv, cheap 5x5 depthwise branch, and shuffle."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        if c2 < 2 or c2 % 2:
            raise ValueError(f"GSConv requires a positive even output width, got c2={c2}.")
        hidden_channels = c2 // 2
        self.cv1 = Conv(c1, hidden_channels, k, s, p, g, d, act)
        self.cv2 = Conv(
            hidden_channels,
            hidden_channels,
            5,
            1,
            2,
            hidden_channels,
            d,
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate primary and cheap features, then interleave their channels."""

        primary = self.cv1(x)
        features = torch.cat((primary, self.cv2(primary)), dim=1)
        features = features.reshape(
            features.shape[0],
            2,
            features.shape[1] // 2,
            features.shape[2],
            features.shape[3],
        )
        features = features.permute(0, 2, 1, 3, 4)
        return features.reshape(
            features.shape[0],
            -1,
            features.shape[3],
            features.shape[4],
        )


class PartialPConv(nn.Module):
    """Official FasterNet ``Partial_conv3`` using its training-safe split_cat path."""

    def __init__(self, channels: int, n_div: int = 4) -> None:
        super().__init__()
        if n_div != 4:
            raise ValueError(f"The controlled experiment requires n_div=4, got {n_div}.")
        self.dim_conv3 = channels // n_div
        self.dim_untouched = channels - self.dim_conv3
        if self.dim_conv3 < 1:
            raise ValueError(f"PartialPConv requires at least {n_div} channels, got {channels}.")
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3,
            self.dim_conv3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3x3 convolution to one quarter and concatenate untouched channels."""

        convolved, untouched = torch.split(
            x,
            [self.dim_conv3, self.dim_untouched],
            dim=1,
        )
        convolved = self.partial_conv3(convolved)
        return torch.cat((convolved, untouched), dim=1)


class PartialPConvProjection(nn.Module):
    """Official PConv spatial mixer plus the minimal pointwise channel projection."""

    def __init__(self, c1: int, c2: int, n_div: int = 4) -> None:
        super().__init__()
        self.spatial_mixer = PartialPConv(c1, n_div=n_div)
        self.project = Conv(c1, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix one-quarter of the channels spatially, then project to ``c2``."""

        return self.project(self.spatial_mixer(x))


class PartialPConvDown(nn.Module):
    """Official PConv mixer followed by a 1x1 stride-2 projection adapter."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        n_div: int = 4,
    ) -> None:
        super().__init__()
        if k != 3:
            raise ValueError(f"PartialPConv's official spatial mixer requires k=3, got {k}.")
        if s != 2:
            raise ValueError(f"PartialPConvDown requires stride 2, got {s}.")
        self.spatial_mixer = PartialPConv(c1, n_div=n_div)
        self.project_down = Conv(c1, c2, k=1, s=s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix at the input resolution, then project and downsample uniformly."""

        return self.project_down(self.spatial_mixer(x))


class DWSeparableConv(nn.Module):
    """Ultralytics depthwise convolution followed by a pointwise Conv."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.dw = DWConv(c1, c1, k=k, s=s, d=d, act=act)
        self.pw = Conv(c1, c2, k=1, s=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise spatial filtering and pointwise channel projection."""

        return self.pw(self.dw(x))


class _LiteBottleneck(nn.Module):
    """Keep Bottleneck.cv1 and its residual rule; scope the experiment to cv2."""

    operator_factory: Callable[[int, int], nn.Module]

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        if g != 1:
            raise ValueError("The controlled DPLS lightweight experiments require g=1.")
        if tuple(k) != (3, 3):
            raise ValueError(f"The controlled experiment requires k=(3, 3), got {k}.")
        hidden_channels = int(c2 * e)
        if hidden_channels < 1:
            raise ValueError(f"Invalid hidden width from c2={c2}, e={e}.")

        # Identical to Ultralytics 8.4.92 Bottleneck.cv1.
        self.cv1 = Conv(c1, hidden_channels, k[0], 1)
        self.cv2 = self.operator_factory(hidden_channels, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the unchanged shortcut rule around the selected cv2 operator."""

        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class _GSConvBottleneck(_LiteBottleneck):
    operator_factory = staticmethod(lambda c1, c2: GSConv(c1, c2, k=3, s=1))


class _GhostConvBottleneck(_LiteBottleneck):
    operator_factory = staticmethod(lambda c1, c2: GhostConv(c1, c2, k=3, s=1))


class _PartialPConvBottleneck(_LiteBottleneck):
    operator_factory = staticmethod(
        lambda c1, c2: PartialPConvProjection(c1, c2, n_div=4)
    )


class _DWSeparableBottleneck(_LiteBottleneck):
    operator_factory = staticmethod(
        lambda c1, c2: DWSeparableConv(c1, c2, k=3, s=1)
    )


class _DWConvBottleneck(_LiteBottleneck):
    operator_factory = staticmethod(lambda c1, c2: DWConv(c1, c2, k=3, s=1))


class _C3k2LiteVariant(C2f):
    """C3k2-compatible container limited to the controlled non-C3k path."""

    bottleneck_type: type[_LiteBottleneck]

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        if c3k:
            raise ValueError(f"{type(self).__name__} requires c3k=False.")
        if attn:
            raise ValueError(f"{type(self).__name__} does not add attention.")
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            # C3k2 uses Bottleneck's default e=0.5 on this path.
            self.bottleneck_type(self.c, self.c, shortcut=shortcut, g=g, e=0.5)
            for _ in range(n)
        )


class C3k2_GSConvLite(_C3k2LiteVariant):
    """C3k2 with only Bottleneck.cv2 replaced by original GSConv."""

    bottleneck_type = _GSConvBottleneck


class C3k2_GhostConvLite(_C3k2LiteVariant):
    """C3k2 with only Bottleneck.cv2 replaced by Ultralytics GhostConv."""

    bottleneck_type = _GhostConvBottleneck


class C3k2_PartialPConvLite(_C3k2LiteVariant):
    """C3k2 with official PConv mixing plus minimal pointwise projection in cv2."""

    bottleneck_type = _PartialPConvBottleneck


class C3k2_DWSeparableLite(_C3k2LiteVariant):
    """C3k2 with only Bottleneck.cv2 replaced by depthwise-pointwise Conv."""

    bottleneck_type = _DWSeparableBottleneck


class C3k2_DWConvLite(_C3k2LiteVariant):
    """C3k2 with only Bottleneck.cv2 replaced by bare Ultralytics DWConv."""

    bottleneck_type = _DWConvBottleneck
