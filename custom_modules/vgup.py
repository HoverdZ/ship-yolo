"""Visibility-Gated Unified Processing (VGUP) for ship detection.

BPW and KBL are implemented based on ERUP-YOLO, WACV 2025.

VGUP retains the unified differentiable processing concept of ERUP-YOLO and
introduces:
1. a residual acceptance gate for BPW,
2. a spatial visibility gate for KBL,
3. a lightweight parameter encoder for remote-sensing ship detection.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from custom_modules.erup import (
    BPWFilter,
    BPW_PARAMETER_COUNT,
    KBLFilter,
    KBL_PARAMETER_COUNT,
    _require_image,
    _tensor_stats,
)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 spatial convolution followed by a pointwise projection."""

    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(c1, c1, 3, stride, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(image))


class LightweightVGUPEncoder(nn.Module):
    """One lightweight encoder shared by filters and both VGUP gates."""

    parameter_output_count = (
        BPW_PARAMETER_COUNT + KBL_PARAMETER_COUNT
    )

    def __init__(
        self,
        in_channels: int = 3,
        prediction_size: int = 128,
        use_global_gate: bool = True,
        use_spatial_gate: bool = True,
        global_gate_bias: float = -1.5,
        spatial_gate_bias: float = -1.0,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("LightweightVGUPEncoder requires RGB input.")
        if prediction_size < 32:
            raise ValueError("prediction_size must be at least 32.")
        self.prediction_size = int(prediction_size)
        self.use_global_gate = bool(use_global_gate)
        self.use_spatial_gate = bool(use_spatial_gate)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.filter_head = nn.Linear(
            128,
            self.parameter_output_count,
        )
        self.global_gate_head = (
            nn.Linear(128, 1)
            if self.use_global_gate
            else None
        )
        self.spatial_gate_head = (
            nn.Conv2d(128, 1, 1)
            if self.use_spatial_gate
            else None
        )

        nn.init.normal_(self.filter_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.filter_head.bias)
        if self.global_gate_head is not None:
            nn.init.normal_(
                self.global_gate_head.weight,
                mean=0.0,
                std=1e-4,
            )
            nn.init.constant_(
                self.global_gate_head.bias,
                global_gate_bias,
            )
        if self.spatial_gate_head is not None:
            nn.init.normal_(
                self.spatial_gate_head.weight,
                mean=0.0,
                std=1e-4,
            )
            nn.init.constant_(
                self.spatial_gate_head.bias,
                spatial_gate_bias,
            )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        _require_image(image)
        resized = F.interpolate(
            image,
            size=(self.prediction_size, self.prediction_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.blocks(self.stem(resized))
        pooled = self.pool(features).flatten(1)
        filter_params = self.filter_head(pooled).tanh()
        bpw_params, kbl_params = torch.split(
            filter_params,
            (BPW_PARAMETER_COUNT, KBL_PARAMETER_COUNT),
            dim=1,
        )
        if self.global_gate_head is None:
            global_gate = image.new_ones((image.shape[0], 1, 1, 1))
        else:
            global_gate = self.global_gate_head(pooled).sigmoid().view(
                image.shape[0],
                1,
                1,
                1,
            )
        if self.spatial_gate_head is None:
            spatial_gate_lowres = image.new_ones(
                (image.shape[0], 1, features.shape[-2], features.shape[-1])
            )
            spatial_gate = image.new_ones(
                (image.shape[0], 1, image.shape[-2], image.shape[-1])
            )
        else:
            spatial_gate_lowres = self.spatial_gate_head(features).sigmoid()
            spatial_gate = F.interpolate(
                spatial_gate_lowres,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return {
            "bpw_params": bpw_params,
            "kbl_params": kbl_params,
            "global_gate": global_gate,
            "spatial_gate": spatial_gate,
            "spatial_gate_lowres": spatial_gate_lowres,
        }


class VGUPPreprocessor(nn.Module):
    """Complete VGUP: lightweight encoder plus both residual gates."""

    out_channels = 3

    def __init__(
        self,
        in_channels: int = 3,
        bpw_segments: int = 8,
        prediction_size: int = 128,
        use_global_gate: bool = True,
        use_spatial_gate: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("VGUPPreprocessor must be the first RGB model layer.")
        self.in_channels = in_channels
        self.use_global_gate = bool(use_global_gate)
        self.use_spatial_gate = bool(use_spatial_gate)
        self.encoder = LightweightVGUPEncoder(
            in_channels=in_channels,
            prediction_size=prediction_size,
            use_global_gate=self.use_global_gate,
            use_spatial_gate=self.use_spatial_gate,
        )
        self.bpw = BPWFilter(segments=bpw_segments)
        self.kbl = KBLFilter()

    @staticmethod
    def apply_bpw_gate(
        image: torch.Tensor,
        enhanced: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return image + gate * (enhanced - image)

    @staticmethod
    def apply_kbl_gate(
        image_bpw: torch.Tensor,
        enhanced: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return image_bpw + gate * (enhanced - image_bpw)

    def forward(
        self,
        image: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        predictions = self.encoder(image)
        enhanced_bpw = self.bpw(image, predictions["bpw_params"])
        gated_bpw = self.apply_bpw_gate(
            image,
            enhanced_bpw,
            predictions["global_gate"],
        )
        enhanced_kbl = self.kbl(
            gated_bpw,
            predictions["kbl_params"],
        )
        output = self.apply_kbl_gate(
            gated_bpw,
            enhanced_kbl,
            predictions["spatial_gate"],
        )
        if not return_debug:
            return output
        global_gate = predictions["global_gate"]
        spatial_gate = predictions["spatial_gate"]
        debug = {
            **predictions,
            "bpw_param_stats": _tensor_stats(predictions["bpw_params"]),
            "kbl_param_stats": _tensor_stats(predictions["kbl_params"]),
            "global_gate_stats": _tensor_stats(global_gate),
            "spatial_gate_stats": _tensor_stats(spatial_gate),
            "bpw_image": enhanced_bpw,
            "gated_bpw_image": gated_bpw,
            "kbl_image": enhanced_kbl,
            "output_image": output,
        }
        return output, debug


__all__ = [
    "DepthwiseSeparableConv",
    "LightweightVGUPEncoder",
    "VGUPPreprocessor",
]
