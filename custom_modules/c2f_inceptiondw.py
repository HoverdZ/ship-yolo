"""YOLOv8 C2f variant with scoped InceptionDW spatial extraction."""

from __future__ import annotations

from torch import nn
from ultralytics.nn.modules import C2f

from custom_modules.c3k2_inceptiondw import InceptionDWBottleneck


class C2f_InceptionDW(C2f):
    """Keep native YOLOv8 C2f topology and replace only Bottleneck ``cv2``.

    Ultralytics YOLOv8 C2f constructs each internal Bottleneck with ``e=1.0``.
    Keeping that expansion value preserves the original first 3x3 convolution
    and shortcut geometry.  Only the second 3x3 spatial operator is replaced
    by the channel-preserving InceptionDW implementation.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            InceptionDWBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                e=1.0,
            )
            for _ in range(n)
        )
