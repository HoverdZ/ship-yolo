"""Controlled Dense/DS allocation variants for the LDPP regression tower."""

from __future__ import annotations

import copy
from collections.abc import Sequence

from torch import nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

from custom_modules.ldpp_detect import _UltralyticsDepthwiseSeparableBlock


class _LDPPAllocationStudyBase(Detect):
    """Share frozen LDPP behavior while varying only cv2 spatial allocation."""

    regression_layout: tuple[tuple[str, str], ...] = ()

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
            raise ValueError(f"LDPP allocation study requires reg_max=16, got {reg_max!r}.")
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
            raise ValueError(f"Detection channels must be positive integers, got {ch!r}.")
        if len(self.regression_layout) != 3 or any(
            len(scale_layout) != 2 or any(block not in {"dense", "ds"} for block in scale_layout)
            for scale_layout in self.regression_layout
        ):
            raise RuntimeError(f"Invalid fixed regression layout: {self.regression_layout!r}.")

        channels = tuple(ch)
        effective_end2end = bool(end2end)
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=effective_end2end,
            ch=channels,
        )

        reg_hidden_channels = max(16, channels[0] // 4, self.reg_max * 4)
        self.reg_hidden_channels = reg_hidden_channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                self._make_spatial_block(layout[0], c1, reg_hidden_channels),
                self._make_spatial_block(
                    layout[1], reg_hidden_channels, reg_hidden_channels
                ),
                nn.Conv2d(reg_hidden_channels, 4 * self.reg_max, kernel_size=1),
            )
            for c1, layout in zip(channels, self.regression_layout)
        )

        if effective_end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.end2end = effective_end2end

    @staticmethod
    def _make_spatial_block(kind: str, c1: int, c2: int) -> nn.Module:
        """Build the frozen Dense or canonical depthwise-separable block."""

        if kind == "dense":
            return Conv(c1, c2, k=3, s=1)
        return _UltralyticsDepthwiseSeparableBlock(c1, c2)


class LDPPDetectH0AllDS(_LDPPAllocationStudyBase):
    """H0: DS+DS at P2, P3 and P4."""

    regression_layout = (("ds", "ds"), ("ds", "ds"), ("ds", "ds"))


class LDPPDetectH1P2Dense(_LDPPAllocationStudyBase):
    """H1: Dense+DS at P2; DS+DS at P3 and P4."""

    regression_layout = (("dense", "ds"), ("ds", "ds"), ("ds", "ds"))


class LDPPDetectH3AllScaleDenseFirst(_LDPPAllocationStudyBase):
    """H3: Dense+DS at every detection scale."""

    regression_layout = (("dense", "ds"),) * 3


class LDPPDetectH4AllDense(_LDPPAllocationStudyBase):
    """H4: Dense+Dense at every detection scale."""

    regression_layout = (("dense", "dense"),) * 3


class LDPPDetectH5P23DenseSecond(_LDPPAllocationStudyBase):
    """H5: DS+Dense at P2/P3; DS+DS at P4."""

    regression_layout = (("ds", "dense"), ("ds", "dense"), ("ds", "ds"))


__all__ = [
    "LDPPDetectH0AllDS",
    "LDPPDetectH1P2Dense",
    "LDPPDetectH3AllScaleDenseFirst",
    "LDPPDetectH4AllDense",
    "LDPPDetectH5P23DenseSecond",
]
