"""C3k2-compatible CrossConv blocks for YOLO11.

The spatial operators reuse the CrossConv construction already present in
Ultralytics 8.4.92's ``C3x`` implementation: ``1 x k`` followed by ``k x 1``.
"""

from __future__ import annotations

from torch import nn
from ultralytics.nn.modules import Bottleneck, C2f, C3x


class C3k2CrossConv(C2f):
    """Preserve C3k2/C2f topology while replacing its spatial units with CrossConv."""

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
        if attn:
            raise ValueError("C3k2CrossConv does not add attention.")
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.c3k = bool(c3k)
        self.m = nn.ModuleList(
            C3x(self.c, self.c, n=2, shortcut=shortcut, g=g, e=0.5)
            if self.c3k
            else Bottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=((1, 3), (3, 1)),
                e=1.0,
            )
            for _ in range(n)
        )
