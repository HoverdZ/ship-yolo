"""Controlled detail-extractor variants of CGDR for YOLO11.

Only the operation between ``detail_reduce`` and ``detail_proj`` differs from
the original CGDR. The HHSPP context path, context gate, zero-initialized
detail scale, and residual fusion are intentionally kept identical.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

from custom_modules.hhspp import HHSPP


class _CentralDifferenceKernel(nn.Module):
    """DEA-Net central-difference kernel branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def get_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.conv.weight
        flat = weight.flatten(2)
        difference = weight.new_zeros(flat.shape)
        difference.copy_(flat)
        difference[:, :, 4] = flat[:, :, 4] - flat.sum(dim=2)
        return difference.view_as(weight), self.conv.bias


class _HorizontalDifferenceKernel(nn.Module):
    """DEA-Net horizontal-difference kernel branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1, bias=True)

    def get_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.conv.weight
        difference = weight.new_zeros(weight.shape[0], weight.shape[1], 9)
        difference[:, :, [0, 3, 6]] = weight
        difference[:, :, [2, 5, 8]] = -weight
        return difference.view(weight.shape[0], weight.shape[1], 3, 3), self.conv.bias


class _VerticalDifferenceKernel(nn.Module):
    """DEA-Net vertical-difference kernel branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1, bias=True)

    def get_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.conv.weight
        difference = weight.new_zeros(weight.shape[0], weight.shape[1], 9)
        difference[:, :, [0, 1, 2]] = weight
        difference[:, :, [6, 7, 8]] = -weight
        return difference.view(weight.shape[0], weight.shape[1], 3, 3), self.conv.bias


class _AngularDifferenceKernel(nn.Module):
    """DEA-Net angular-difference kernel branch."""

    _CLOCKWISE_PERMUTATION = (3, 0, 1, 6, 4, 2, 7, 8, 5)

    def __init__(self, channels: int, theta: float = 1.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.theta = float(theta)

    def get_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.conv.weight
        flat = weight.flatten(2)
        difference = flat - self.theta * flat[:, :, self._CLOCKWISE_PERMUTATION]
        return difference.view_as(weight), self.conv.bias


class _DEConv(nn.Module):
    """DEA-Net DEConv with device- and dtype-safe kernel construction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.central = _CentralDifferenceKernel(channels)
        self.horizontal = _HorizontalDifferenceKernel(channels)
        self.vertical = _VerticalDifferenceKernel(channels)
        self.angular = _AngularDifferenceKernel(channels)
        self.vanilla = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        central_weight, central_bias = self.central.get_weight()
        horizontal_weight, horizontal_bias = self.horizontal.get_weight()
        vertical_weight, vertical_bias = self.vertical.get_weight()
        angular_weight, angular_bias = self.angular.get_weight()
        weight = (
            central_weight
            + horizontal_weight
            + vertical_weight
            + angular_weight
            + self.vanilla.weight
        )
        bias = (
            central_bias
            + horizontal_bias
            + vertical_bias
            + angular_bias
            + self.vanilla.bias
        )
        return F.conv2d(x, weight, bias=bias, stride=1, padding=1)


class _PixelDifferenceDepthwiseConv2d(nn.Conv2d):
    """PiDiNet pixel-difference definitions specialized to a 3x3 DWConv."""

    _CLOCKWISE_PERMUTATION = (3, 0, 1, 6, 4, 2, 7, 8, 5)
    _RADIAL_POSITIVE = (0, 2, 4, 10, 14, 20, 22, 24)
    _RADIAL_NEGATIVE = (6, 7, 8, 11, 13, 16, 17, 18)

    def __init__(self, channels: int, operation: str) -> None:
        if operation not in {"cv", "cd", "ad", "rd"}:
            raise ValueError(f"Unsupported PDC operation: {operation!r}.")
        super().__init__(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=channels,
            bias=False,
        )
        self.operation = operation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.operation == "cv":
            return F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=1,
                padding=1,
                dilation=1,
                groups=self.groups,
            )

        if self.operation == "cd":
            center_weight = self.weight.sum(dim=(2, 3), keepdim=True)
            center = F.conv2d(
                x,
                center_weight,
                stride=1,
                padding=0,
                groups=self.groups,
            )
            vanilla = F.conv2d(
                x,
                self.weight,
                bias=self.bias,
                stride=1,
                padding=1,
                dilation=1,
                groups=self.groups,
            )
            return vanilla - center

        flat = self.weight.flatten(2)
        if self.operation == "ad":
            angular_weight = flat - flat[:, :, self._CLOCKWISE_PERMUTATION]
            return F.conv2d(
                x,
                angular_weight.view_as(self.weight),
                bias=self.bias,
                stride=1,
                padding=1,
                dilation=1,
                groups=self.groups,
            )

        radial_weight = self.weight.new_zeros(
            self.weight.shape[0], self.weight.shape[1], 25
        )
        radial_weight[:, :, self._RADIAL_POSITIVE] = flat[:, :, 1:]
        radial_weight[:, :, self._RADIAL_NEGATIVE] = -flat[:, :, 1:]
        radial_weight[:, :, 12] = 0
        return F.conv2d(
            x,
            radial_weight.view(self.weight.shape[0], self.weight.shape[1], 5, 5),
            bias=self.bias,
            stride=1,
            padding=2,
            dilation=1,
            groups=self.groups,
        )


class _PDCBlock(nn.Module):
    """Shape-preserving PiDiNet PDCBlock."""

    def __init__(self, channels: int, operation: str) -> None:
        super().__init__()
        self.conv1 = _PixelDifferenceDepthwiseConv2d(channels, operation)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        return y + x


class _PDCDetailExtractor(nn.Module):
    """One local CD -> AD -> RD -> CV cycle from PiDiNet ``carv4``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cd = _PDCBlock(channels, "cd")
        self.ad = _PDCBlock(channels, "ad")
        self.rd = _PDCBlock(channels, "rd")
        self.cv = _PDCBlock(channels, "cv")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cd(x)
        x = self.ad(x)
        x = self.rd(x)
        return self.cv(x)


def _contrast_kernel(kernel_size: int, inner_size: int = 1) -> torch.Tensor:
    """Reproduce HDNet ``GenerateKernels`` for one inner-window scale."""

    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError(f"MAC kernel size must be odd and greater than one, got {kernel_size}.")
    if inner_size <= 0 or inner_size % 2 == 0 or inner_size >= kernel_size:
        raise ValueError(
            "MAC inner size must be positive, odd, and smaller than the kernel."
        )
    outer_count = kernel_size * kernel_size - inner_size * inner_size
    kernel = torch.full(
        (kernel_size, kernel_size),
        fill_value=-1.0 / outer_count,
        dtype=torch.float32,
    )
    offset = (kernel_size - inner_size) // 2
    kernel[offset : offset + inner_size, offset : offset + inner_size] = (
        1.0 / (inner_size * inner_size)
    )
    return kernel


def _initialize_depthwise_kernel(conv: nn.Conv2d, kernel: torch.Tensor) -> None:
    expected_shape = conv.weight.shape[-2:]
    if tuple(kernel.shape) != tuple(expected_shape):
        raise ValueError(
            f"MAC initialization kernel {tuple(kernel.shape)} does not match "
            f"convolution kernel {tuple(expected_shape)}."
        )
    with torch.no_grad():
        prepared = kernel.to(device=conv.weight.device, dtype=conv.weight.dtype)
        conv.weight.copy_(prepared.view(1, 1, *expected_shape).expand_as(conv.weight))


class _MAC(nn.Module):
    """HDNet multi-scale atrous contrast block for equal input/output channels."""

    def __init__(self, channels: int, scales: int = 4) -> None:
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"MAC requires positive integer channels, got {channels!r}.")
        if scales != 4:
            raise ValueError(f"HDNet MAC requires exactly four scales, got {scales!r}.")
        if channels % scales != 0:
            raise ValueError(
                f"MAC channels ({channels}) must be divisible by scales ({scales})."
            )

        self.scales = scales
        self.spx = channels // scales
        self.relu = nn.ReLU(inplace=True)
        self.inconv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.spx,
                self.spx,
                3,
                stride=1,
                padding=1,
                groups=self.spx,
            ),
            nn.BatchNorm2d(self.spx),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                self.spx,
                self.spx,
                3,
                stride=1,
                padding=2,
                dilation=2,
                groups=self.spx,
            ),
            nn.BatchNorm2d(self.spx),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                self.spx,
                self.spx,
                3,
                stride=1,
                padding=1,
                groups=self.spx,
            )
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                self.spx,
                self.spx,
                3,
                stride=1,
                padding=2,
                dilation=2,
                groups=self.spx,
            )
        )
        self.conv5 = nn.Sequential(nn.BatchNorm2d(self.spx))
        self.outconv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        contrast = _contrast_kernel(3)
        average = torch.full((3, 3), 1.0 / 9.0, dtype=torch.float32)
        surround = torch.full((3, 3), -1.0 / 8.0, dtype=torch.float32)
        surround[1, 1] = 0.0
        _initialize_depthwise_kernel(self.conv1[0], contrast)
        _initialize_depthwise_kernel(self.conv2[0], contrast)
        _initialize_depthwise_kernel(self.conv3[0], average)
        _initialize_depthwise_kernel(self.conv4[0], surround)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inconv(x)
        residual = x
        xs = torch.chunk(x, self.scales, dim=1)
        ys = [xs[0]]
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        propagated = xs[3] + ys[2]
        last = self.conv5(self.conv3(propagated) + self.conv4(propagated))
        ys.append(self.relu(last))
        y = self.outconv(torch.cat(ys, dim=1))
        return self.relu(y + residual)


class _CGDRDetailVariant(HHSPP):
    """Exact CGDR outer framework shared only by the new controlled variants."""

    def __init__(
        self,
        c1: int,
        c2: int,
        extractor_factory: Callable[[int], nn.Module],
    ) -> None:
        if not isinstance(c1, int) or isinstance(c1, bool) or c1 <= 0:
            raise ValueError(f"CGDR requires positive integer c1, got {c1!r}.")
        if not isinstance(c2, int) or isinstance(c2, bool) or c2 <= 0:
            raise ValueError(f"CGDR requires positive integer c2, got {c2!r}.")
        super().__init__(c1, c2)
        detail_channels = max(16, c1 // 4)
        if not isinstance(detail_channels, int):
            raise TypeError(
                f"CGDR requires integer channels, got {detail_channels!r}."
            )

        self.detail_channels = detail_channels
        self.detail_reduce = Conv(c1, detail_channels, k=1, s=1)
        self.detail_extractor = extractor_factory(detail_channels)
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
        detail_feature = self.detail_extractor(z)
        detail = self.detail_proj(detail_feature)
        gate = torch.sigmoid(self.detail_gate(context))
        return context + self.detail_scale * gate * detail


class CGDRDEConv(_CGDRDetailVariant):
    """CGDR with the DEA-Net DEConv detail extractor."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__(c1, c2, _DEConv)


class CGDRPDC(_CGDRDetailVariant):
    """CGDR with one PiDiNet carv4 PDC cycle as its detail extractor."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__(c1, c2, _PDCDetailExtractor)


class CGDRMAC(_CGDRDetailVariant):
    """CGDR with the HDNet MAC detail extractor."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__(c1, c2, _MAC)


__all__ = ["CGDRDEConv", "CGDRPDC", "CGDRMAC"]
