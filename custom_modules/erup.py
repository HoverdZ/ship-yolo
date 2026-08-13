"""Differentiable ERUP-YOLO image-adaptive filters and preprocessor.

BPW and KBL are implemented based on ERUP-YOLO, WACV 2025:
"ERUP-YOLO: Enhancing Object Detection Robustness for Adverse Weather
Condition by Unified Image-Adaptive Processing".

No author-maintained official implementation was available when this module
was written. The equations follow the paper and supplementary material; the
engineering choices not fixed by the paper are documented in
The parameterization follows the implementation used by ``VGUPPreprocessor``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

BPW_PARAMETER_COUNT = 12
KBL_KERNEL_SIZE = 9
KBL_PARAMETER_COUNT = 2 * 3 * KBL_KERNEL_SIZE * KBL_KERNEL_SIZE
ERUP_PARAMETER_COUNT = BPW_PARAMETER_COUNT + KBL_PARAMETER_COUNT


def _require_image(image: torch.Tensor) -> None:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            "Expected an RGB BCHW tensor with shape [B,3,H,W], "
            f"got {tuple(image.shape)}."
        )
    if not image.is_floating_point():
        raise TypeError("Adaptive image filters require a floating-point tensor.")


def _require_params(
    params: torch.Tensor,
    *,
    batch: int,
    count: int,
    name: str,
) -> None:
    if params.shape != (batch, count):
        raise ValueError(
            f"{name} parameters must have shape [{batch},{count}], "
            f"got {tuple(params.shape)}."
        )
    if not params.is_floating_point():
        raise TypeError(f"{name} parameters must be floating point.")


def _tensor_stats(value: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "mean": value.mean(),
        "minimum": value.amin(),
        "maximum": value.amax(),
    }


class BPWFilter(nn.Module):
    """Cubic Bezier curve-based pixel-wise filter from ERUP-YOLO.

    Parameters are ordered per RGB channel as ``r1, theta1, r2, theta2`` and
    are expected in ``[-1, 1]``. Equation (12) is evaluated as a differentiable
    piecewise-linear approximation over ``segments`` intervals.
    """

    parameter_count = BPW_PARAMETER_COUNT

    def __init__(self, segments: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        if not isinstance(segments, int) or segments < 2:
            raise ValueError(f"segments must be an integer >= 2, got {segments!r}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.segments = segments
        self.eps = float(eps)
        self.register_buffer(
            "q",
            torch.linspace(0.0, 1.0, segments + 1).view(1, 1, -1),
            persistent=False,
        )

    @staticmethod
    def control_points(params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Map Eq. (13)/(14) parameters to the two 2D control points."""

        r1, theta1, r2, theta2 = params.unbind(dim=-1)
        radius1 = (r1 + 1.0) * 0.5
        radius2 = (r2 + 1.0) * 0.5
        angle1 = (theta1 + 1.0) * (math.pi / 4.0)
        angle2 = (theta2 + 1.0) * (math.pi / 4.0)
        p1x = radius1 * torch.cos(angle1)
        p1y = radius1 * torch.sin(angle1)
        p2x = 1.0 - radius2 * torch.cos(angle2)
        p2y = 1.0 - radius2 * torch.sin(angle2)
        return p1x, p1y, p2x, p2y

    def forward(
        self,
        image: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        _require_image(image)
        batch = image.shape[0]
        _require_params(
            params,
            batch=batch,
            count=self.parameter_count,
            name="BPW",
        )
        channel_params = params.view(batch, 3, 4).clamp(-1.0, 1.0)
        p1x, p1y, p2x, p2y = self.control_points(channel_params)
        q = self.q.to(dtype=image.dtype, device=image.device)
        one_minus_q = 1.0 - q

        def curve(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            first = first.unsqueeze(-1)
            second = second.unsqueeze(-1)
            return (
                3.0 * q * one_minus_q.square() * first
                + 3.0 * q.square() * one_minus_q * second
                + q.pow(3)
            )

        curve_x = curve(p1x, p2x)
        curve_y = curve(p1y, p2y)
        delta_x = torch.diff(curve_x, dim=-1).clamp_min(self.eps)
        delta_y = torch.diff(curve_y, dim=-1)
        lower_x = curve_x[..., :-1]

        pixels = image.clamp(0.0, 1.0).unsqueeze(2)
        progress = (pixels - lower_x[..., None, None]).clamp_min(0.0)
        progress = torch.minimum(
            progress,
            delta_x[..., None, None],
        )
        output = (
            progress
            * (delta_y / delta_x)[..., None, None]
        ).sum(dim=2)
        return output.clamp(0.0, 1.0)


class KBLFilter(nn.Module):
    """Kernel-based local filter from ERUP-YOLO Eq. (15)."""

    kernel_size = KBL_KERNEL_SIZE
    parameter_count = KBL_PARAMETER_COUNT

    def __init__(self, kernel_size: int = KBL_KERNEL_SIZE) -> None:
        super().__init__()
        if kernel_size != KBL_KERNEL_SIZE:
            raise ValueError(
                "The first ERUP reproduction must retain the paper's 9x9 kernels."
            )
        self.padding = kernel_size // 2

    def _dynamic_depthwise(
        self,
        image: torch.Tensor,
        kernels: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = image.shape
        grouped_image = image.reshape(1, batch * channels, height, width)
        grouped_kernels = kernels.reshape(
            batch * channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )
        output = F.conv2d(
            grouped_image,
            grouped_kernels,
            padding=self.padding,
            groups=batch * channels,
        )
        return output.view(batch, channels, height, width)

    def forward(
        self,
        image: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        _require_image(image)
        batch = image.shape[0]
        _require_params(
            params,
            batch=batch,
            count=self.parameter_count,
            name="KBL",
        )
        kernels = params.view(
            batch,
            2,
            3,
            self.kernel_size,
            self.kernel_size,
        ).clamp(-1.0, 1.0)
        response1 = self._dynamic_depthwise(image, kernels[:, 0])
        response2 = self._dynamic_depthwise(image, kernels[:, 1])
        return image * response1 + response2 + image


class ERUPParameterEncoder(nn.Module):
    """Paper-level reproduction of the original wide ERUP parameter encoder."""

    output_count = ERUP_PARAMETER_COUNT

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("ERUPParameterEncoder requires three RGB channels.")
        channels = (64, 128, 256, 512, 1024)
        blocks: list[nn.Module] = []
        current = in_channels
        for output in channels:
            blocks.extend(
                (
                    nn.Conv2d(current, output, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(3, 2, 1),
                )
            )
            current = output
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(channels[-1], self.output_count)
        # The paper does not publish initialization. Small weights and zero
        # bias keep sigmoid outputs close to 0.5 (near-identity filters) while
        # still allowing gradients to reach every encoder stage on step one.
        nn.init.normal_(self.projection.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.projection.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _require_image(image)
        features = self.features(image)
        logits = self.projection(self.pool(features).flatten(1))
        return 2.0 * logits.sigmoid() - 1.0


class ERUPPreprocessor(nn.Module):
    """Original ERUP BPW -> KBL preprocessing path."""

    out_channels = 3
    parameter_count = ERUP_PARAMETER_COUNT

    def __init__(
        self,
        in_channels: int = 3,
        bpw_segments: int = 8,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("ERUPPreprocessor must be the first RGB model layer.")
        self.in_channels = in_channels
        self.encoder = ERUPParameterEncoder(in_channels)
        self.bpw = BPWFilter(segments=bpw_segments)
        self.kbl = KBLFilter()

    def forward(
        self,
        image: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        params = self.encoder(image)
        bpw_params, kbl_params = torch.split(
            params,
            (BPW_PARAMETER_COUNT, KBL_PARAMETER_COUNT),
            dim=1,
        )
        image_bpw = self.bpw(image, bpw_params)
        image_out = self.kbl(image_bpw, kbl_params)
        if not return_debug:
            return image_out
        debug = {
            "bpw_params": bpw_params,
            "kbl_params": kbl_params,
            "bpw_param_stats": _tensor_stats(bpw_params),
            "kbl_param_stats": _tensor_stats(kbl_params),
            "bpw_image": image_bpw,
            "output_image": image_out,
        }
        return image_out, debug


__all__ = [
    "BPWFilter",
    "BPW_PARAMETER_COUNT",
    "ERUPParameterEncoder",
    "ERUPPreprocessor",
    "ERUP_PARAMETER_COUNT",
    "KBLFilter",
    "KBL_KERNEL_SIZE",
    "KBL_PARAMETER_COUNT",
]
