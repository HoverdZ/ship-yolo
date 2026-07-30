"""Task-adapted calibration of the original FFCA-YOLO SCAM.

SCAM is adapted from FFCA-YOLO:
FFCA-YOLO for Small Object Detection in Remote Sensing Images,
IEEE Transactions on Geoscience and Remote Sensing, 2024.
Official repository:
https://github.com/yemu1138178251/FFCA-YOLO

CA-SCAM retains the original SCAM context residual and introduces
a zero-initialized local-contrast-conditioned residual calibration
for low-contrast and cluttered remote-sensing ship features.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from custom_modules.scam import SCAM


class _ContrastCalibratedSCAMBase(SCAM):
    """Shared local-contrast branch for controlled CA-SCAM ablations."""

    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels)
        self.local_mean = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        self.contrast_proj = nn.Conv2d(
            1,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        # sigmoid(0) = 0.5.  Zero initialization makes every learnable
        # calibration variant deterministic at construction time.
        nn.init.zeros_(self.contrast_proj.weight)
        nn.init.zeros_(self.contrast_proj.bias)

    def contrast_map(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw local contrast and its learnable spatial map."""

        local_mean = self.local_mean(x)
        local_contrast = torch.abs(x - local_mean).mean(
            dim=1,
            keepdim=True,
        )
        contrast_map = self.contrast_proj(local_contrast).sigmoid()
        return local_contrast, contrast_map

    def calibration_beta(self) -> torch.Tensor:
        """Return the variant-specific residual calibration strength."""

        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        delta = self.compute_context_residual(x)
        local_contrast, contrast_map = self.contrast_map(x)
        beta = self.calibration_beta()
        output = x + (1.0 + beta * contrast_map) * delta
        if not return_debug:
            return output
        return output, {
            "beta": beta,
            "local_contrast": local_contrast,
            "contrast_map": contrast_map,
            "context_residual": delta,
            "calibrated_residual": (1.0 + beta * contrast_map) * delta,
        }


class CASCAMFixedBeta(_ContrastCalibratedSCAMBase):
    """CA-SCAM ablation with the contrast branch and a fixed beta."""

    def __init__(
        self,
        in_channels: int,
        fixed_beta: float = 0.1,
    ) -> None:
        if not 0.0 < fixed_beta <= 1.0:
            raise ValueError(
                f"fixed_beta must be in (0, 1], got {fixed_beta}."
            )
        super().__init__(in_channels)
        self.register_buffer(
            "fixed_beta",
            torch.tensor(float(fixed_beta)),
            persistent=True,
        )

    def calibration_beta(self) -> torch.Tensor:
        return self.fixed_beta


class CASCAMUnbounded(_ContrastCalibratedSCAMBase):
    """CA-SCAM ablation with a learnable, intentionally unbounded beta."""

    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels)
        # beta=0 preserves exact SCAM output at initialization.  This
        # controlled ablation intentionally omits the final tanh bound.
        self.contrast_beta = nn.Parameter(torch.zeros(1))

    def calibration_beta(self) -> torch.Tensor:
        return self.contrast_beta


class CASCAM(_ContrastCalibratedSCAMBase):
    """Contrast-Aware SCAM with bounded, equivalent-initialized calibration."""

    def __init__(
        self,
        in_channels: int,
        max_delta: float = 0.1,
    ) -> None:
        if not 0.0 < max_delta <= 1.0:
            raise ValueError(
                f"max_delta must be in (0, 1], got {max_delta}."
            )
        super().__init__(in_channels)
        self.contrast_logit = nn.Parameter(torch.zeros(1))
        self.max_delta = float(max_delta)

    def calibration_beta(self) -> torch.Tensor:
        return self.max_delta * torch.tanh(self.contrast_logit)

    def contrast_state(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw local contrast, its spatial map, and bounded beta."""

        local_contrast, contrast_map = self.contrast_map(x)
        beta = self.calibration_beta()
        return local_contrast, contrast_map, beta


__all__ = [
    "CASCAM",
    "CASCAMFixedBeta",
    "CASCAMUnbounded",
]
