"""Lightweight Detect tower candidates for the frozen LDPP neck.

Only feature-tower structure changes across the YOLOX, RTMDet, and hybrid
candidates. The prediction protocol, DFL representation, anchor generation,
decoding, loss, export path, and optional end-to-end path remain those of
Ultralytics Detect.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


_BN_EPS = 1e-3
_BN_MOMENTUM = 0.03


def _validate_detect_args(
    nc: int,
    width: int,
    width_name: str,
    reg_max: int,
    end2end: bool | None,
    ch: Sequence[int],
) -> tuple[tuple[int, int, int], bool]:
    """Validate the fixed three-scale YOLO11 Detect contract."""

    if not isinstance(nc, int) or isinstance(nc, bool) or nc < 1:
        raise ValueError(f"nc must be a positive integer, got {nc!r}.")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError(f"{width_name} must be a positive integer, got {width!r}.")
    if not isinstance(reg_max, int) or isinstance(reg_max, bool) or reg_max != 16:
        raise ValueError(f"These lightweight heads require reg_max=16, got {reg_max!r}.")
    if end2end is not None and not isinstance(end2end, bool):
        raise TypeError(f"end2end must be a boolean or None, got {type(end2end).__name__}.")
    if not isinstance(ch, (list, tuple)) or len(ch) != 3:
        raise ValueError("Expected exactly three ordered detection channels: P2, P3, P4.")
    if any(
        not isinstance(channel, int) or isinstance(channel, bool) or channel < 1
        for channel in ch
    ):
        raise ValueError(f"Detection input channels must be positive integers, got {ch!r}.")
    return tuple(ch), bool(end2end)


class _ConvBNAct(nn.Sequential):
    """Conv-BN-SiLU block with YOLOX/RTMDet normalization defaults."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
            nn.SiLU(inplace=False),
        )


class _DepthwiseSeparableBlock(nn.Sequential):
    """Depthwise 3x3 then pointwise 1x1, each followed by BN and SiLU."""

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        out_channels = in_channels if out_channels is None else out_channels
        super().__init__(
            _ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                groups=in_channels,
            ),
            _ConvBNAct(in_channels, out_channels, kernel_size=1),
        )


class _UltralyticsDepthwiseSeparableBlock(nn.Sequential):
    """Ultralytics-native depthwise 3x3 followed by pointwise 1x1."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            DWConv(in_channels, in_channels, k=3, s=1),
            Conv(in_channels, out_channels, k=1, s=1),
        )


class _ProjectedDetect(Detect):
    """Detect protocol adapter for a common per-level projection before task split."""

    @property
    def one2many(self) -> dict[str, nn.Module]:
        """Return one-to-many components, including the per-level projections."""

        return dict(box_head=self.cv2, cls_head=self.cv3, stem_head=self.stems)

    @property
    def one2one(self) -> dict[str, nn.Module]:
        """Return independently copied one-to-one components."""

        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            stem_head=self.one2one_stems,
        )

    def _configure_end2end(self, enabled: bool) -> None:
        """Create an independent one-to-one copy after installing custom towers."""

        if enabled:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_stems = copy.deepcopy(self.stems)
        self.end2end = enabled

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.ModuleList | None = None,
        cls_head: nn.ModuleList | None = None,
        stem_head: nn.ModuleList | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Apply each projection once, then return the official Detect mapping."""

        if box_head is None or cls_head is None or stem_head is None:
            return {}
        if len(x) != self.nl:
            raise ValueError(f"Expected {self.nl} ordered detection features, got {len(x)}.")

        batch_size = x[0].shape[0]
        projected = [stem_head[index](x[index]) for index in range(self.nl)]
        boxes = torch.cat(
            [
                box_head[index](projected[index]).view(
                    batch_size, 4 * self.reg_max, -1
                )
                for index in range(self.nl)
            ],
            dim=-1,
        )
        scores = torch.cat(
            [
                cls_head[index](projected[index]).view(batch_size, self.nc, -1)
                for index in range(self.nl)
            ],
            dim=-1,
        )
        return dict(boxes=boxes, scores=scores, feats=x)


class YOLOXNanoDWDetect(_ProjectedDetect):
    """YOLOX-Nano-style depthwise decoupled towers with YOLO11 outputs."""

    def __init__(
        self,
        nc: int = 80,
        hidden_channels: int = 64,
        reg_max: int = 16,
        end2end: bool | None = False,
        ch: Sequence[int] = (),
    ) -> None:
        channels, effective_end2end = _validate_detect_args(
            nc, hidden_channels, "hidden_channels", reg_max, end2end, ch
        )
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=channels)

        self.hidden_channels = hidden_channels
        self.depthwise = True
        self.tower_depth = 2
        self.stems = nn.ModuleList(
            _ConvBNAct(channel, hidden_channels, kernel_size=1) for channel in channels
        )
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                _DepthwiseSeparableBlock(hidden_channels),
                _DepthwiseSeparableBlock(hidden_channels),
                nn.Conv2d(hidden_channels, 4 * self.reg_max, kernel_size=1),
            )
            for _ in channels
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                _DepthwiseSeparableBlock(hidden_channels),
                _DepthwiseSeparableBlock(hidden_channels),
                nn.Conv2d(hidden_channels, self.nc, kernel_size=1),
            )
            for _ in channels
        )
        self._configure_end2end(effective_end2end)


class YOLO11ClsYOLOXNanoDWRegDetect(Detect):
    """Keep stock YOLO11 classification and lighten only regression features.

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
            raise ValueError(
                "YOLO11ClsYOLOXNanoDWRegDetect requires "
                f"reg_max=16, got {reg_max!r}."
            )
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
            nn.Sequential(
                _DepthwiseSeparableBlock(channel, reg_hidden_channels),
                _DepthwiseSeparableBlock(
                    reg_hidden_channels,
                    reg_hidden_channels,
                ),
                nn.Conv2d(
                    reg_hidden_channels,
                    4 * self.reg_max,
                    kernel_size=1,
                ),
            )
            for channel in channels
        )

        if effective_end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.end2end = effective_end2end


class _YOLO11ClsHybridDWRegDetect(Detect):
    """Keep stock classification while selectively restoring dense regression layers."""

    _dense_first_levels: tuple[int, ...] = ()

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
            raise ValueError(
                f"{type(self).__name__} requires reg_max=16, got {reg_max!r}."
            )
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
            nn.Sequential(
                Conv(channel, reg_hidden_channels, k=3, s=1)
                if level_index in self._dense_first_levels
                else _UltralyticsDepthwiseSeparableBlock(
                    channel,
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
            )
            for level_index, channel in enumerate(channels)
        )

        if effective_end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.end2end = effective_end2end


class YOLO11ClsP2DenseHybridDWRegDetect(_YOLO11ClsHybridDWRegDetect):
    """Use Dense+DS at P2 and DS+DS at P3/P4 for regression."""

    _dense_first_levels = (0,)

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool | None = False,
        ch: Sequence[int] = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)


class YOLO11ClsP23DenseHybridDWRegDetect(_YOLO11ClsHybridDWRegDetect):
    """Use Dense+DS at P2/P3 and DS+DS at P4 for regression."""

    _dense_first_levels = (0, 1)

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool | None = False,
        ch: Sequence[int] = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)


class _SharedSepBNTower(nn.Module):
    """RTMDet depthwise tower with shared DW/PW weights and per-level BN."""

    def __init__(self, channels: int, stacked_convs: int, num_levels: int) -> None:
        super().__init__()
        self.stacked_convs = stacked_convs
        self.num_levels = num_levels
        self.depthwise_convs = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            )
            for _ in range(stacked_convs)
        )
        self.pointwise_convs = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            for _ in range(stacked_convs)
        )
        self.depthwise_norms = nn.ModuleList(
            nn.ModuleList(
                nn.BatchNorm2d(channels, eps=_BN_EPS, momentum=_BN_MOMENTUM)
                for _ in range(num_levels)
            )
            for _ in range(stacked_convs)
        )
        self.pointwise_norms = nn.ModuleList(
            nn.ModuleList(
                nn.BatchNorm2d(channels, eps=_BN_EPS, momentum=_BN_MOMENTUM)
                for _ in range(num_levels)
            )
            for _ in range(stacked_convs)
        )
        self.activation = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor, level_index: int) -> torch.Tensor:
        """Run shared convolutions through the BN pair owned by one level."""

        if not 0 <= level_index < self.num_levels:
            raise IndexError(
                f"level_index must be in [0, {self.num_levels}), got {level_index}."
            )
        for stage in range(self.stacked_convs):
            x = self.activation(
                self.depthwise_norms[stage][level_index](self.depthwise_convs[stage](x))
            )
            x = self.activation(
                self.pointwise_norms[stage][level_index](self.pointwise_convs[stage](x))
            )
        return x


class RTMDetSepBNLiteDetect(_ProjectedDetect):
    """RTMDet SepBN depthwise towers with YOLO11 DFL predictions."""

    def __init__(
        self,
        nc: int = 80,
        feat_channels: int = 64,
        stacked_convs: int = 2,
        reg_max: int = 16,
        end2end: bool | None = False,
        ch: Sequence[int] = (),
    ) -> None:
        channels, effective_end2end = _validate_detect_args(
            nc, feat_channels, "feat_channels", reg_max, end2end, ch
        )
        if (
            not isinstance(stacked_convs, int)
            or isinstance(stacked_convs, bool)
            or stacked_convs < 1
        ):
            raise ValueError(
                f"stacked_convs must be a positive integer, got {stacked_convs!r}."
            )
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=channels)

        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.share_conv = True
        self.use_depthwise = True
        self.stems = nn.ModuleList(
            _ConvBNAct(channel, feat_channels, kernel_size=1) for channel in channels
        )
        self.reg_tower = _SharedSepBNTower(feat_channels, stacked_convs, self.nl)
        self.cls_tower = _SharedSepBNTower(feat_channels, stacked_convs, self.nl)
        self.cv2 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(feat_channels, 4 * self.reg_max, kernel_size=1))
            for _ in channels
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(feat_channels, self.nc, kernel_size=1))
            for _ in channels
        )

        if effective_end2end:
            self.one2one_reg_tower = copy.deepcopy(self.reg_tower)
            self.one2one_cls_tower = copy.deepcopy(self.cls_tower)
        self._configure_end2end(effective_end2end)

    @property
    def one2many(self) -> dict[str, nn.Module]:
        """Return standard components plus the two independent SepBN towers."""

        components = super().one2many
        components.update(box_tower=self.reg_tower, cls_tower=self.cls_tower)
        return components

    @property
    def one2one(self) -> dict[str, nn.Module]:
        """Return independently copied end-to-end SepBN components."""

        components = super().one2one
        components.update(
            box_tower=self.one2one_reg_tower,
            cls_tower=self.one2one_cls_tower,
        )
        return components

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.ModuleList | None = None,
        cls_head: nn.ModuleList | None = None,
        stem_head: nn.ModuleList | None = None,
        box_tower: _SharedSepBNTower | None = None,
        cls_tower: _SharedSepBNTower | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run shared-conv/separate-BN towers and emit Detect-format tensors."""

        if any(
            component is None
            for component in (box_head, cls_head, stem_head, box_tower, cls_tower)
        ):
            return {}
        if len(x) != self.nl:
            raise ValueError(f"Expected {self.nl} ordered detection features, got {len(x)}.")

        batch_size = x[0].shape[0]
        box_outputs = []
        cls_outputs = []
        for level_index in range(self.nl):
            projected = stem_head[level_index](x[level_index])
            box_outputs.append(
                box_head[level_index](box_tower(projected, level_index))
            )
            cls_outputs.append(
                cls_head[level_index](cls_tower(projected, level_index))
            )
        boxes = torch.cat(
            [
                output.view(batch_size, 4 * self.reg_max, -1)
                for output in box_outputs
            ],
            dim=-1,
        )
        scores = torch.cat(
            [output.view(batch_size, self.nc, -1) for output in cls_outputs],
            dim=-1,
        )
        return dict(boxes=boxes, scores=scores, feats=x)
