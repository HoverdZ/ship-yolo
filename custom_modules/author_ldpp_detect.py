"""LDPP regression adapter bound to the active pinned YOLOv12/v13 author fork.

Use author_fork_registration in a fresh process before constructing a model.
This module must not be imported by the standard Ultralytics registration.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import ultralytics
from torch import nn
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


if (
    ultralytics.__version__ != "8.3.63"
    or tuple(inspect.signature(Detect.__init__).parameters) != ("self", "nc", "ch")
):
    raise RuntimeError(
        "AuthorLDPPDetect requires a pinned YOLOv12/v13 author fork; "
        "use its registration helper in a fresh process, not standard Ultralytics."
    )


class _AuthorDepthwiseSeparableBlock(nn.Sequential):
    """Native depthwise 3x3 then pointwise 1x1, without an extra stem."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__(DWConv(c1, c1, k=3, s=1), Conv(c1, c2, k=1, s=1))


class AuthorLDPPDetect(Detect):
    """Keep author classification/DFL/decode; adapt only scale-aware regression.

    The author parser appends only channels to YAML [nc], so ch must be
    the second argument. Inputs are ordered P2/P3/P4 or P2/P3/P4/P5.
    """

    def __init__(self, nc: int = 80, ch: Sequence[int] = ()) -> None:
        if not isinstance(nc, int) or isinstance(nc, bool) or nc < 1:
            raise ValueError(f"nc must be a positive integer, got {nc!r}.")
        if not isinstance(ch, (list, tuple)) or len(ch) not in (3, 4):
            raise ValueError("Expected ordered P2/P3/P4 or P2/P3/P4/P5 channels.")
        if any(not isinstance(c, int) or isinstance(c, bool) or c < 1 for c in ch):
            raise ValueError(f"Detection channels must be positive integers, got {ch!r}.")
        if self.end2end:
            raise ValueError("The pinned author LDPP experiment requires end2end=False.")

        channels = tuple(ch)
        super().__init__(nc=nc, ch=channels)
        if self.reg_max != 16:
            raise RuntimeError("The pinned author Detect must use reg_max=16.")
        c2 = max(16, channels[0] // 4, self.reg_max * 4)
        self.reg_hidden_channels = c2
        # Leave parent-created cv3 untouched, including its legacy selection.
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(c1, c2, k=3, s=1) if i < 2
                else _AuthorDepthwiseSeparableBlock(c1, c2),
                _AuthorDepthwiseSeparableBlock(c2, c2),
                nn.Conv2d(c2, 4 * self.reg_max, kernel_size=1),
            )
            for i, c1 in enumerate(channels)
        )


__all__ = ["AuthorLDPPDetect"]
