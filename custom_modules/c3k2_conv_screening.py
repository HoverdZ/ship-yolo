"""Controlled C3k2 variants for shallow small-object convolution screening.

Every variant preserves the standard first 3x3 convolution and shortcut in
each YOLO11 Bottleneck. Only Bottleneck.cv2 is replaced. The containing YAMLs
use these modules exclusively at backbone layers 2 and 4 (P2/P3).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from ultralytics.nn.modules import C2f, Conv

from custom_modules.lsk_conv import LargeSelectiveKernelConvBNAct
from custom_modules.pinwheel_conv import PinwheelConv


class _ScopedConvBottleneck(nn.Module):
    """Preserve YOLO Bottleneck.cv1 and replace only its second convolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        operator_factory: Callable[[int, int], nn.Module],
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        if g != 1:
            raise ValueError("The controlled P2/P3 experiments require g=1.")
        if tuple(k) != (3, 3):
            raise ValueError(f"The controlled experiments require k=(3, 3), got {k}.")
        hidden_channels = int(c2 * e)
        if hidden_channels <= 0:
            raise ValueError(f"Invalid hidden channel count from c2={c2}, e={e}.")

        # Identical to Ultralytics 8.4.92 Bottleneck.cv1.
        self.cv1 = Conv(c1, hidden_channels, k[0], 1)
        self.cv2 = operator_factory(hidden_channels, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the original residual rule around the selected cv2 operator."""

        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


def _channel_preserving_adapter(
    c1: int,
    c2: int,
    operator: nn.Module,
) -> nn.Sequential:
    """Adapt cv1 hidden channels before a channel-preserving official core."""

    adapter: nn.Module = (
        nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        if c1 != c2
        else nn.Identity()
    )
    return nn.Sequential(adapter, operator)


class PConvBottleneck(_ScopedConvBottleneck):
    """YOLO Bottleneck whose cv2 is the official pinwheel convolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__(
            c1,
            c2,
            lambda hidden, output: PinwheelConv(hidden, output, kernel_size=3, stride=1),
            shortcut=shortcut,
            g=g,
            k=k,
            e=e,
        )


class LSKConvBottleneck(_ScopedConvBottleneck):
    """YOLO Bottleneck whose cv2 is the LSK spatial-selection core."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__(
            c1,
            c2,
            lambda hidden, output: _channel_preserving_adapter(
                hidden,
                output,
                LargeSelectiveKernelConvBNAct(output),
            ),
            shortcut=shortcut,
            g=g,
            k=k,
            e=e,
        )


class _C3k2ConvVariant(C2f):
    """C3k2-compatible container restricted to non-C3k shallow blocks."""

    bottleneck_type: type[_ScopedConvBottleneck]

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
            raise ValueError(f"{type(self).__name__} is limited to c3k=False P2/P3 blocks.")
        if attn:
            raise ValueError(f"{type(self).__name__} does not add C2PSA attention.")
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            # C3k2 (unlike its C2f parent) reconstructs these Bottlenecks with
            # the Bottleneck default e=0.5. Matching that value preserves cv1
            # exactly and leaves only cv2 as the experimental variable.
            self.bottleneck_type(self.c, self.c, shortcut=shortcut, g=g, e=0.5)
            for _ in range(n)
        )


class C3k2_PConv(_C3k2ConvVariant):
    """C3k2 using PConv only for the second convolution."""

    bottleneck_type = PConvBottleneck


class C3k2_LSKConv(_C3k2ConvVariant):
    """C3k2 using LSK only for the second convolution."""

    bottleneck_type = LSKConvBottleneck
