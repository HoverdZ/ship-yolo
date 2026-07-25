"""DySample adapted from the official ICCV 2023 implementation.

Paper: Learning to Upsample by Learning to Sample (ICCV 2023)
Official repository: https://github.com/tiny-smart/dysample
Reference commit: 81a1de5caa95d55a0f5488425fa53ec7ef47f8f0
License: MIT

The offset prediction, initialization, pixel shuffle, and ``grid_sample``
mathematics are kept faithful. This adapter adds explicit validation and a
constructor whose first argument is the input channel count supplied by the
Ultralytics YAML parser.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _normal_init(
    module: nn.Module,
    mean: float = 0.0,
    std: float = 1.0,
    bias: float = 0.0,
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _constant_init(
    module: nn.Module,
    value: float,
    bias: float = 0.0,
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, value)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """Official DySample operator with channel-preserving YOLO integration."""

    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if not isinstance(scale, int) or scale <= 0:
            raise ValueError(f"scale must be a positive integer, got {scale!r}.")
        if style not in {"lp", "pl"}:
            raise ValueError(f"style must be 'lp' or 'pl', got {style!r}.")
        if not isinstance(groups, int) or groups <= 0:
            raise ValueError(f"groups must be a positive integer, got {groups!r}.")
        if in_channels < groups or in_channels % groups != 0:
            raise ValueError(
                "DySample requires in_channels to be divisible by groups; "
                f"got in_channels={in_channels}, groups={groups}."
            )
        if style == "pl" and (
            in_channels < scale**2 or in_channels % scale**2 != 0
        ):
            raise ValueError(
                "DySample style='pl' requires in_channels to be divisible by "
                f"scale**2; got in_channels={in_channels}, scale={scale}."
            )

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.scale = scale
        self.style = style
        self.groups = groups
        self.dyscope = bool(dyscope)

        offset_in_channels = (
            in_channels // scale**2 if style == "pl" else in_channels
        )
        offset_out_channels = (
            2 * groups if style == "pl" else 2 * groups * scale**2
        )

        self.offset = nn.Conv2d(offset_in_channels, offset_out_channels, 1)
        _normal_init(self.offset, std=0.001)
        if self.dyscope:
            self.scope = nn.Conv2d(
                offset_in_channels,
                offset_out_channels,
                1,
                bias=False,
            )
            _constant_init(self.scope, value=0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        h = (
            torch.arange(
                (-self.scale + 1) / 2,
                (self.scale - 1) / 2 + 1,
            )
            / self.scale
        )
        mesh = torch.meshgrid(h, h, indexing="ij")
        return (
            torch.stack(mesh)
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)

        coords_h = torch.arange(height, device=x.device) + 0.5
        coords_w = torch.arange(width, device=x.device) + 0.5
        coords = (
            torch.stack(
                torch.meshgrid(coords_w, coords_h, indexing="ij")
            )
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .to(dtype=x.dtype, device=x.device)
        )
        normalizer = torch.tensor(
            [width, height],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(
                coords.view(batch, -1, height, width),
                self.scale,
            )
            .view(
                batch,
                2,
                -1,
                self.scale * height,
                self.scale * width,
            )
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(
                batch * self.groups,
                -1,
                height,
                width,
            ),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(
            batch,
            -1,
            self.scale * height,
            self.scale * width,
        )

    def forward_lp(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "scope"):
            offset = (
                self.offset(x) * self.scope(x).sigmoid() * 0.5
                + self.init_pos
            )
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x: torch.Tensor) -> torch.Tensor:
        shuffled = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = (
                F.pixel_unshuffle(
                    self.offset(shuffled) * self.scope(shuffled).sigmoid(),
                    self.scale,
                )
                * 0.5
                + self.init_pos
            )
        else:
            offset = (
                F.pixel_unshuffle(
                    self.offset(shuffled),
                    self.scale,
                )
                * 0.25
                + self.init_pos
            )
        return self.sample(x, offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"DySample expects a BCHW tensor, got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"DySample expected {self.in_channels} channels, "
                f"got {x.shape[1]}."
            )
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)
