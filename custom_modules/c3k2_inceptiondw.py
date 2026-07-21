"""YOLO11 C3k2 variant with InceptionDW only in the P2/P3 bottleneck cv2."""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import C2f, Conv

from custom_modules.inceptiondw import InceptionDWConvBNAct


class InceptionDWBottleneck(nn.Module):
    """Preserve the YOLO11 bottleneck cv1/residual and replace cv2 spatial work."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        if g != 1:
            raise ValueError("The scoped YOLO11n P2/P3 experiment requires g=1.")
        hidden_channels = int(c2 * e)
        if hidden_channels <= 0:
            raise ValueError(f"Invalid hidden channel count from c2={c2}, e={e}.")

        # This is intentionally identical to Ultralytics 8.4.92 Bottleneck.cv1.
        self.cv1 = Conv(c1, hidden_channels, k[0], 1)

        # Ultralytics C3k2 uses Bottleneck e=0.5, so the original cv2 also
        # expands hidden_channels back to c2. The official InceptionDW core is
        # channel preserving; this unactivated, unnormalized 1x1 performs only
        # that required channel adaptation before the replacement spatial op.
        self.cv2_adapter: nn.Module = (
            nn.Conv2d(hidden_channels, c2, kernel_size=1, stride=1, bias=False)
            if hidden_channels != c2
            else nn.Identity()
        )
        self.cv2 = InceptionDWConvBNAct(
            c2,
            square_kernel_size=square_kernel_size,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
        )
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bottleneck with the original shortcut condition."""

        y = self.cv2(self.cv2_adapter(self.cv1(x)))
        return x + y if self.add else y


class C3k2_InceptionDW(C2f):
    """C3k2-compatible module scoped to the two non-C3k YOLO11n backbone blocks."""

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
            raise ValueError("C3k2_InceptionDW is intentionally limited to c3k=False P2/P3 blocks.")
        if attn:
            raise ValueError("C3k2_InceptionDW does not add attention.")
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            InceptionDWBottleneck(self.c, self.c, shortcut=shortcut, g=g)
            for _ in range(n)
        )
