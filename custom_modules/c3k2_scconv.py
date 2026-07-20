"""Ultralytics 8.4.92 C3k2 variants with internal 3x3 Conv replaced by ScConv."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from ultralytics.nn.modules.block import C2f, C3, PSABlock
from ultralytics.nn.modules.conv import Conv

from custom_modules.scconv import ScConv


class _ScConvBNAct(nn.Module):
    """Replace only Conv2d while retaining Ultralytics Conv's BN and activation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        scconv_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.scconv = ScConv(c1, out_channels=c2, **dict(scconv_kwargs or {}))
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SCConv, then the original Ultralytics BN and SiLU."""

        return self.act(self.bn(self.scconv(x)))


class SCBottleneck(nn.Module):
    """YOLO bottleneck whose two 3x3 Conv2d layers use ScConv."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        scconv_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        if hidden <= 0:
            raise ValueError(
                f"SCBottleneck expansion creates zero hidden channels: "
                f"c2={c2}, e={e}, hidden={hidden}."
            )
        if tuple(k) != (3, 3):
            raise ValueError(f"SCBottleneck only replaces 3x3 convolutions, got k={k}.")
        if not isinstance(g, int) or isinstance(g, bool) or g <= 0:
            raise ValueError(f"g must be a positive integer, got {g!r}.")

        first_kwargs = dict(scconv_kwargs or {})
        second_kwargs = dict(first_kwargs)
        if g > 1:
            requested_group_size = second_kwargs.get("group_size", g)
            if requested_group_size != g:
                raise ValueError(
                    f"Conflicting group settings: Bottleneck g={g}, "
                    f"ScConv group_size={requested_group_size}."
                )
            second_kwargs["group_size"] = g

        self.cv1 = _ScConvBNAct(c1, hidden, first_kwargs)
        self.cv2 = _ScConvBNAct(hidden, c2, second_kwargs)
        self.add = shortcut and c1 == c2
        self.groups = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two SCConv blocks and the unchanged optional shortcut."""

        transformed = self.cv2(self.cv1(x))
        return x + transformed if self.add else transformed


class C3k_SCConv(C3):
    """C3k topology with only its internal Bottleneck 3x3 layers replaced."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
        scconv_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if k != 3:
            raise ValueError(f"C3k_SCConv targets the 3x3 C3k path only, got k={k}.")
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        self.m = nn.Sequential(
            *(
                SCBottleneck(
                    hidden,
                    hidden,
                    shortcut=shortcut,
                    g=g,
                    k=(k, k),
                    e=1.0,
                    scconv_kwargs=scconv_kwargs,
                )
                for _ in range(n)
            )
        )


class C3k2_SCConv(C2f):
    """Ultralytics C3k2 topology with internal 3x3 spatial Conv replaced.

    Constructor order matches Ultralytics 8.4.92 ``C3k2`` so ``parse_model``
    can inject scaled ``c1``, ``c2``, and repeat count ``n`` identically.
    """

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
        scconv_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                SCBottleneck(
                    self.c,
                    self.c,
                    shortcut=shortcut,
                    g=g,
                    e=0.5,
                    scconv_kwargs=scconv_kwargs,
                ),
                PSABlock(
                    self.c,
                    attn_ratio=0.5,
                    num_heads=max(self.c // 64, 1),
                ),
            )
            if attn
            else C3k_SCConv(
                self.c,
                self.c,
                2,
                shortcut,
                g,
                scconv_kwargs=scconv_kwargs,
            )
            if c3k
            else SCBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                e=0.5,
                scconv_kwargs=scconv_kwargs,
            )
            for _ in range(n)
        )
