"""Bounded semantic confirmation and concat for the YOLO P3 top-down fusion."""

from __future__ import annotations

import torch
from torch import nn


class SemanticConfirmationGate(nn.Module):
    """Use top-down P4 semantics to modulate C3 and perform the native concat.

    This module replaces the original concat layer and returns
    ``cat(upsampled_p4, modulated_c3)``. Subsequent layer indices, shapes, and
    pretrained parameter keys therefore stay unchanged. The final mask is
    zero-initialized, making the replacement equivalent to native concat at
    initialization.
    """

    def __init__(
        self,
        channels: list[int] | tuple[int, int],
        hidden_ratio: float = 0.25,
        alpha_max: float = 0.25,
    ) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError("SemanticConfirmationGate requires [C3, upsampled P4].")
        c3_channels, semantic_channels = map(int, channels)
        if c3_channels <= 0 or semantic_channels <= 0:
            raise ValueError(f"Invalid input channels: {channels}.")
        if not 0.0 < hidden_ratio <= 1.0:
            raise ValueError("hidden_ratio must be in (0, 1].")
        if not 0.0 < alpha_max < 1.0:
            raise ValueError("alpha_max must be in (0, 1).")

        hidden = max(8, int(c3_channels * hidden_ratio))
        self.c3_channels = c3_channels
        self.semantic_channels = semantic_channels
        self.alpha_max = float(alpha_max)
        self.local_projection = nn.Sequential(
            nn.Conv2d(c3_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        )
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(semantic_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        )
        self.mask = nn.Conv2d(hidden, 1, 3, padding=1, bias=True)
        nn.init.zeros_(self.mask.weight)
        nn.init.zeros_(self.mask.bias)

    def modulation(self, c3: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        if c3.shape[-2:] != semantic.shape[-2:]:
            raise ValueError(
                "C3 and upsampled P4 must share spatial size, got "
                f"{tuple(c3.shape[-2:])} and {tuple(semantic.shape[-2:])}."
            )
        interaction = self.local_projection(c3) * self.semantic_projection(semantic)
        gate = torch.sigmoid(self.mask(interaction))
        return 1.0 + self.alpha_max * (2.0 * gate - 1.0)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError("SemanticConfirmationGate expects [C3, upsampled P4].")
        c3, semantic = inputs
        if c3.ndim != 4 or semantic.ndim != 4:
            raise ValueError("SemanticConfirmationGate inputs must be BCHW tensors.")
        if c3.shape[1] != self.c3_channels or semantic.shape[1] != self.semantic_channels:
            raise ValueError(
                f"Expected channels [{self.c3_channels}, {self.semantic_channels}], "
                f"got [{c3.shape[1]}, {semantic.shape[1]}]."
            )
        modulated_c3 = c3 * self.modulation(c3, semantic)
        return torch.cat((semantic, modulated_c3), dim=1)
