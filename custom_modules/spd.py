"""Targeted space-to-depth downsampling for small-object feature preservation."""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules import Conv


class SPDDown(nn.Module):
    """Space-to-depth by 2 followed by a non-strided Ultralytics Conv.

    The phase order is explicit and stable so a stride-2 3x3 convolution can
    be mapped exactly into the non-strided 3x3 convolution for pretrained
    initialization.
    """

    phase_order = ((0, 0), (1, 0), (0, 1), (1, 1))

    def __init__(self, c1: int, c2: int, k: int = 3) -> None:
        super().__init__()
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"SPDDown channels must be positive, got c1={c1}, c2={c2}.")
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"SPDDown kernel must be a positive odd integer, got {k}.")
        self.c1 = c1
        self.c2 = c2
        self.factor = 2
        self.cv = Conv(4 * c1, c2, k=k, s=1)

    def space_to_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange four spatial phases into channels without discarding pixels."""

        if x.ndim != 4:
            raise ValueError(f"SPDDown expects BCHW input, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.c1:
            raise ValueError(f"SPDDown expected {self.c1} channels, got {x.shape[1]}.")
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "SPDDown requires even spatial dimensions; "
                f"got HxW={x.shape[-2]}x{x.shape[-1]}."
            )
        return torch.cat(
            [x[..., row::2, col::2] for row, col in self.phase_order],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by rearrangement and a stride-1 convolution."""

        return self.cv(self.space_to_depth(x))
