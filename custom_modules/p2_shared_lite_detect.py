"""P2-only task-shared lightweight Detect head for the DPLS neck."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from ultralytics.nn.modules import Conv, DWConv
from ultralytics.nn.modules.head import Detect


class P2SharedLiteDetect(Detect):
    """Share one lightweight P2 stem between box and classification branches.

    P3 and P4 keep the exact towers created by Ultralytics 8.4.92 ``Detect``.
    The constructor order intentionally matches this repository's parse-model
    adaptation: YAML arguments ``[nc, p2_hidden]`` are followed by
    ``[reg_max, end2end, ch]``.
    """

    def __init__(
        self,
        nc: int = 80,
        p2_hidden: int = 48,
        reg_max: int = 16,
        end2end: bool | None = False,
        ch: Sequence[int] = (),
    ) -> None:
        if not isinstance(nc, int) or isinstance(nc, bool) or nc < 1:
            raise ValueError(f"nc must be a positive integer, got {nc!r}.")
        if (
            not isinstance(p2_hidden, int)
            or isinstance(p2_hidden, bool)
            or p2_hidden < 1
        ):
            raise ValueError(
                f"p2_hidden must be a positive integer, got {p2_hidden!r}."
            )
        if not isinstance(reg_max, int) or isinstance(reg_max, bool) or reg_max != 16:
            raise ValueError(f"P2SharedLiteDetect requires reg_max=16, got {reg_max!r}.")
        if end2end is not None and not isinstance(end2end, bool):
            raise TypeError(f"end2end must be a boolean, got {type(end2end).__name__}.")
        if end2end:
            raise ValueError(
                "P2SharedLiteDetect supports only the standard one-to-many YOLO11 path."
            )
        if not isinstance(ch, (list, tuple)) or len(ch) != 3:
            raise ValueError(
                "P2SharedLiteDetect requires exactly three ordered channels: P2, P3, P4."
            )
        if any(
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or channel < 1
            for channel in ch
        ):
            raise ValueError(f"Detection input channels must be positive integers, got {ch}.")
        channels = tuple(ch)

        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=channels)
        self.p2_hidden = p2_hidden

        self.p2_shared_stem = nn.Sequential(
            Conv(channels[0], p2_hidden, k=1, s=1),
            DWConv(p2_hidden, p2_hidden, k=3, s=1),
            Conv(p2_hidden, p2_hidden, k=1, s=1),
        )
        self.cv2[0] = nn.Sequential(
            Conv(p2_hidden, p2_hidden, k=1, s=1),
            nn.Conv2d(p2_hidden, 4 * self.reg_max, kernel_size=1),
        )
        self.cv3[0] = nn.Sequential(
            Conv(p2_hidden, p2_hidden, k=1, s=1),
            nn.Conv2d(p2_hidden, self.nc, kernel_size=1),
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.ModuleList | None = None,
        cls_head: nn.ModuleList | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Return official-format predictions after computing the P2 stem once."""

        if box_head is None or cls_head is None:  # Match Detect's fused path.
            return {}
        if len(x) != self.nl:
            raise ValueError(f"Expected {self.nl} ordered detection features, got {len(x)}.")

        batch_size = x[0].shape[0]
        p2_shared = self.p2_shared_stem(x[0])

        box_outputs = [box_head[0](p2_shared)]
        cls_outputs = [cls_head[0](p2_shared)]
        box_outputs.extend(box_head[index](x[index]) for index in range(1, self.nl))
        cls_outputs.extend(cls_head[index](x[index]) for index in range(1, self.nl))

        boxes = torch.cat(
            [output.view(batch_size, 4 * self.reg_max, -1) for output in box_outputs],
            dim=-1,
        )
        scores = torch.cat(
            [output.view(batch_size, self.nc, -1) for output in cls_outputs],
            dim=-1,
        )
        return dict(boxes=boxes, scores=scores, feats=x)
