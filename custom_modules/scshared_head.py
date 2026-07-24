"""Scale-calibrated shared detection head for the YOLO11 ship detector."""

from __future__ import annotations

import math

import torch
from torch import nn

from ultralytics.nn.modules import Detect


def _valid_group_count(channels: int, requested: int) -> int:
    """Return the largest requested-or-smaller GN group count that divides channels."""

    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    """Convolution followed by GroupNorm and SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int,
        *,
        groups: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(_valid_group_count(c2, groups), c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ScaleCalibration(nn.Module):
    """Apply one independently learned, positive scale to a pyramid level."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.log_scale.exp()


class SCSharedDetect(Detect):
    """YOLO11 Detect variant with shared Conv-GN features and level calibration.

    The three prediction levels retain independent input adapters and output
    projections. Only the two central 3x3 blocks are shared. This avoids forcing
    P3/P4/P5 into identical raw feature distributions while still removing most
    repeated head computation. Classification and DFL regression keep separate
    final projections.
    """

    def __init__(
        self,
        nc: int = 80,
        shared_channels: int = 64,
        gn_groups: int = 16,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        if end2end:
            raise ValueError("SCSharedDetect currently supports one-to-many YOLO11 detection only.")
        if len(ch) != 3:
            raise ValueError(f"SCSharedDetect requires [P3, P4, P5], got channels={ch}.")
        if shared_channels < 4 * reg_max:
            raise ValueError(
                "shared_channels must be at least 4 * reg_max so DFL regression "
                f"is not bottlenecked; got {shared_channels} and reg_max={reg_max}."
            )
        if gn_groups <= 0:
            raise ValueError("gn_groups must be positive.")

        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        self.shared_channels = int(shared_channels)
        self.gn_groups = int(gn_groups)

        self.input_adapters = nn.ModuleList(
            ConvGNAct(int(c), self.shared_channels, 1, groups=self.gn_groups)
            for c in ch
        )
        self.shared_stem = nn.Sequential(
            ConvGNAct(
                self.shared_channels,
                self.shared_channels,
                3,
                groups=self.gn_groups,
            ),
            ConvGNAct(
                self.shared_channels,
                self.shared_channels,
                3,
                groups=self.gn_groups,
            ),
        )
        self.scale_calibration = nn.ModuleList(ScaleCalibration() for _ in ch)

        # Keep the standard attribute names so Ultralytics loss/export code can
        # continue to consume the head through Detect.one2many.
        self.cv2 = nn.ModuleList(
            nn.Conv2d(self.shared_channels, 4 * self.reg_max, 1) for _ in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Conv2d(self.shared_channels, self.nc, 1) for _ in ch
        )

    def calibrated_features(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Project, share, and calibrate the three pyramid features."""

        if len(x) != self.nl:
            raise ValueError(f"Expected {self.nl} detection features, got {len(x)}.")
        return [
            self.scale_calibration[i](
                self.shared_stem(self.input_adapters[i](feature))
            )
            for i, feature in enumerate(x)
        ]

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return native YOLO11 DFL boxes and classification logits."""

        if box_head is None or cls_head is None:
            return {}
        shared = self.calibrated_features(x)
        batch_size = x[0].shape[0]
        boxes = torch.cat(
            [
                box_head[i](shared[i]).view(batch_size, 4 * self.reg_max, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        scores = torch.cat(
            [
                cls_head[i](shared[i]).view(batch_size, self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": x}

    def bias_init(self) -> None:
        """Initialize direct box/class output projections after stride discovery."""

        for i, (box_head, cls_head) in enumerate(zip(self.cv2, self.cv3)):
            box_head.bias.data.fill_(2.0)
            cls_head.bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )

    def fuse(self) -> None:
        """The shared one-to-many head has no removable duplicate branch."""

        return None
