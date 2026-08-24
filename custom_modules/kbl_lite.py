"""YOLOv13 host adapter derived from the stable KBL side of VGUP.

The adapter intentionally omits BPW and the global acceptance gate. It keeps
the lightweight image encoder, KBL dynamic filtering and the spatial
visibility gate used by the previously validated YOLOv13 configuration.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from custom_modules.erup import KBLFilter, KBL_PARAMETER_COUNT, _require_image
from custom_modules.vgup import DepthwiseSeparableConv


class LightweightKBLVisibilityEncoder(nn.Module):
    """Predict KBL parameters and one spatial visibility gate."""

    def __init__(
        self,
        in_channels: int = 3,
        prediction_size: int = 128,
        spatial_gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("LightweightKBLVisibilityEncoder requires RGB input.")
        if prediction_size < 32:
            raise ValueError("prediction_size must be at least 32.")
        self.prediction_size = int(prediction_size)

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
        self.kbl_head = nn.Linear(128, KBL_PARAMETER_COUNT)
        self.spatial_gate_head = nn.Conv2d(128, 1, 1)

        nn.init.normal_(self.kbl_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.kbl_head.bias)
        nn.init.normal_(self.spatial_gate_head.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.spatial_gate_head.bias, float(spatial_gate_bias))

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
        kbl_params = self.kbl_head(pooled).tanh()
        spatial_gate_lowres = self.spatial_gate_head(features).sigmoid()
        spatial_gate = F.interpolate(
            spatial_gate_lowres,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return {
            "kbl_params": kbl_params,
            "spatial_gate": spatial_gate,
            "spatial_gate_lowres": spatial_gate_lowres,
        }


class KBLLitePreprocessor(nn.Module):
    """KBL-only VGUP-Lite with a spatial residual visibility gate."""

    out_channels = 3

    def __init__(
        self,
        in_channels: int = 3,
        prediction_size: int = 128,
        spatial_gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("KBLLitePreprocessor must be the first RGB model layer.")
        self.in_channels = int(in_channels)
        self.encoder = LightweightKBLVisibilityEncoder(
            in_channels=in_channels,
            prediction_size=prediction_size,
            spatial_gate_bias=spatial_gate_bias,
        )
        self.kbl = KBLFilter()

    def forward(
        self,
        image: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        _require_image(image)
        predictions = self.encoder(image)
        input_dtype = image.dtype

        if image.device.type in {"cpu", "cuda"}:
            precision_context = torch.autocast(
                device_type=image.device.type,
                enabled=False,
            )
        else:
            precision_context = nullcontext()

        with precision_context:
            image_fp32 = image.float()
            kbl_params_fp32 = predictions["kbl_params"].float()
            spatial_gate_fp32 = predictions["spatial_gate"].float()
            enhanced = self.kbl(image_fp32, kbl_params_fp32)
            output_fp32 = image_fp32 + spatial_gate_fp32 * (
                enhanced - image_fp32
            )
            output_fp32 = output_fp32.clamp(0.0, 1.0)

        if not torch.isfinite(output_fp32).all():
            raise FloatingPointError("KBL-only VGUP-Lite produced NaN/Inf.")

        output = output_fp32.to(dtype=input_dtype)
        if not return_debug:
            return output
        return output, {
            **predictions,
            "kbl_image": enhanced,
            "output_image": output,
        }


__all__ = ["LightweightKBLVisibilityEncoder", "KBLLitePreprocessor"]
