"""Canonical lightweight detail-preserving Detect head for LDPP."""

from __future__ import annotations

import copy
from collections.abc import Sequence

from torch import nn
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


class _UltralyticsDepthwiseSeparableBlock(nn.Sequential):
    """Ultralytics-native depthwise 3x3 followed by pointwise 1x1."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            DWConv(in_channels, in_channels, k=3, s=1),
            Conv(in_channels, out_channels, k=1, s=1),
        )


class LDPPDetect(Detect):
    """Keep stock YOLO11 classification and use the finalized LDPP regression towers.

    YAML arguments ``[nc]`` are followed by Ultralytics 8.4.92 parser values
    ``[reg_max, end2end, ch]``. The parent-created ``cv3`` is intentionally
    retained without replacement.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool | None = False,
        ch: Sequence[int] = (),
    ) -> None:
        if not isinstance(nc, int) or isinstance(nc, bool) or nc < 1:
            raise ValueError(f"nc must be a positive integer, got {nc!r}.")
        if not isinstance(reg_max, int) or isinstance(reg_max, bool) or reg_max != 16:
            raise ValueError(f"LDPPDetect requires reg_max=16, got {reg_max!r}.")
        if end2end is not None and not isinstance(end2end, bool):
            raise TypeError(
                f"end2end must be a boolean or None, got {type(end2end).__name__}."
            )
        if not isinstance(ch, (list, tuple)) or len(ch) != 3:
            raise ValueError("Expected exactly three ordered detection channels: P2, P3, P4.")
        if any(
            not isinstance(channel, int) or isinstance(channel, bool) or channel < 1
            for channel in ch
        ):
            raise ValueError(
                f"Detection input channels must be positive integers, got {ch!r}."
            )

        channels = tuple(ch)
        effective_end2end = bool(end2end)
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=effective_end2end,
            ch=channels,
        )

        reg_hidden_channels = max(
            16,
            channels[0] // 4,
            self.reg_max * 4,
        )
        self.reg_hidden_channels = reg_hidden_channels
        self.cv2 = nn.ModuleList(
            (
                nn.Sequential(
                    Conv(channels[0], reg_hidden_channels, k=3, s=1),
                    _UltralyticsDepthwiseSeparableBlock(
                        reg_hidden_channels,
                        reg_hidden_channels,
                    ),
                    nn.Conv2d(
                        reg_hidden_channels,
                        4 * self.reg_max,
                        kernel_size=1,
                    ),
                ),
                nn.Sequential(
                    Conv(channels[1], reg_hidden_channels, k=3, s=1),
                    _UltralyticsDepthwiseSeparableBlock(
                        reg_hidden_channels,
                        reg_hidden_channels,
                    ),
                    nn.Conv2d(
                        reg_hidden_channels,
                        4 * self.reg_max,
                        kernel_size=1,
                    ),
                ),
                nn.Sequential(
                    _UltralyticsDepthwiseSeparableBlock(
                        channels[2],
                        reg_hidden_channels,
                    ),
                    _UltralyticsDepthwiseSeparableBlock(
                        reg_hidden_channels,
                        reg_hidden_channels,
                    ),
                    nn.Conv2d(
                        reg_hidden_channels,
                        4 * self.reg_max,
                        kernel_size=1,
                    ),
                ),
            )
        )

        if effective_end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.end2end = effective_end2end
