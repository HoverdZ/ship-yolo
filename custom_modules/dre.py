"""Training-only Degraded Reconstruction Enhancer adapted from official DRENet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss


class _DREChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layers(self.pool(x))


class _DREResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            Conv(channels, channels, 3, act=nn.ReLU(inplace=True)),
            Conv(channels, channels, 3),
            _DREChannelAttention(channels, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class _DREResidualGroup(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _DREResidualChannelAttentionBlock(channels, reduction),
            Conv(channels, channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class DREReconstructionEnhancer(nn.Module):
    """One-group, one-block RCAN enhancer used in the official DRENet code."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _DREResidualGroup(channels, reduction),
            Conv(channels, channels, 3),
        )
        self.tail = nn.Sequential(
            Conv(channels, channels * 4, 3),
            nn.PixelShuffle(2),
            Conv(channels, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(x) + x)


def _variable_box_mean(image: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """Compute a truncated-border box mean for every pixel using an integral image."""
    channels, height, width = image.shape
    rows = torch.arange(height, device=image.device).view(height, 1).expand(height, width)
    cols = torch.arange(width, device=image.device).view(1, width).expand(height, width)
    top = (rows - radius).clamp_min(0)
    left = (cols - radius).clamp_min(0)
    bottom = (rows + radius + 1).clamp_max(height)
    right = (cols + radius + 1).clamp_max(width)

    integral = F.pad(image.cumsum(dim=1).cumsum(dim=2), (1, 0, 1, 0))
    flat = integral.reshape(channels, -1)
    integral_width = width + 1

    def gather(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        index = (y * integral_width + x).reshape(1, -1).expand(channels, -1)
        return flat.gather(1, index).reshape(channels, height, width)

    total = gather(bottom, right) - gather(top, right) - gather(bottom, left) + gather(top, left)
    area = ((bottom - top) * (right - left)).clamp_min(1).to(dtype=image.dtype)
    return total / area.unsqueeze(0)


@torch.no_grad()
def selective_degradation_target(
    images: torch.Tensor,
    batch_indices: torch.Tensor,
    bboxes_xywh: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Create DRENet Selective Degradation labels from the synchronized augmented batch.

    Bounding boxes are Ultralytics-normalized ``xywh`` labels. Generation is
    performed in FP32 at input resolution using the author's distance rule,
    then resized by nearest neighbor to the enhancer output resolution, matching
    the official DRENet collate behavior.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"DRE expects BCHW RGB images, got shape {tuple(images.shape)}.")
    images_fp32 = images.detach().float()
    batch_indices = batch_indices.reshape(-1).to(device=images.device, dtype=torch.long)
    bboxes_xywh = bboxes_xywh.reshape(-1, 4).to(device=images.device, dtype=torch.float32)
    _, _, height, width = images_fp32.shape
    rows = torch.arange(height, device=images.device, dtype=torch.float32).view(height, 1)
    cols = torch.arange(width, device=images.device, dtype=torch.float32).view(1, width)
    # The author rule was calibrated at 512x512. Express distances in that
    # reference coordinate system, then scale its blur radius back to the
    # actual tensor size instead of hard-coding a 512-pixel input.
    x_to_reference = 512.0 / width
    y_to_reference = 512.0 / height
    radius_from_reference = (width / 512.0 + height / 512.0) * 0.5
    targets: list[torch.Tensor] = []

    for image_index, image in enumerate(images_fp32):
        boxes = bboxes_xywh[batch_indices == image_index]
        minimum_distance_sq = torch.full(
            (height, width), 130.0**2, device=images.device, dtype=torch.float32
        )
        for center_x, center_y in boxes[:, :2]:
            x = center_x * width
            y = center_y * height
            distance_sq = ((rows - y) * y_to_reference).square() + (
                (cols - x) * x_to_reference
            ).square()
            minimum_distance_sq = torch.minimum(minimum_distance_sq, distance_sq)
        distance = minimum_distance_sq.sqrt().clamp_max(130.0)
        reference_radius = torch.floor(torch.exp(distance * math.log(1.03))).to(torch.long) // 2
        radius = torch.round(reference_radius.float() * radius_from_reference).to(torch.long)
        targets.append(_variable_box_mean(image, radius))

    target = torch.stack(targets, dim=0)
    return F.interpolate(target, size=output_size, mode="nearest")


class DREDetect(Detect):
    """Native Detect plus a training-only reconstruction branch from backbone P3."""

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        if len(ch) != 4:
            raise ValueError(
                "DREDetect requires [reconstruction_source, P3, P4, P5] channels, "
                f"got {len(ch)} inputs."
            )
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=tuple(ch[1:]))
        self.enhancer = DREReconstructionEnhancer(int(ch[0]))
        # s = log(a^2), log(b^2) gives the paper's 1/(2a^2)L + ln(a) form.
        self.dre_log_variances = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.last_reconstruction_loss: torch.Tensor | None = None
        self.training_branch_stripped = False

    def forward(self, x: list[torch.Tensor]):
        reconstruction_source, *detection_features = x
        predictions = super().forward(detection_features)
        if self.training:
            if self.training_branch_stripped:
                raise RuntimeError("A deployment-only DREDetect cannot be returned to training mode.")
            predictions["dre_reconstruction"] = self.enhancer(reconstruction_source)
        return predictions

    def switch_to_deploy(self) -> None:
        """Permanently remove training-only DRE parameters after switching to eval mode."""
        if self.training:
            raise RuntimeError("Call eval() before removing the DRE training branch.")
        if self.training_branch_stripped:
            return
        del self.enhancer
        del self.dre_log_variances
        self.last_reconstruction_loss = None
        self.training_branch_stripped = True


class DREDetectionLoss(v8DetectionLoss):
    """Native YOLO11 detection loss jointly balanced with DRE reconstruction MSE."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        head = model.model[-1]
        if not isinstance(head, DREDetect):
            raise TypeError("DREDetectionLoss requires DREDetect as the final model layer.")
        self.dre_head = head

    def loss(
        self, predictions: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction = predictions.get("dre_reconstruction")
        if reconstruction is None:
            raise RuntimeError("DRE reconstruction output is required while computing the training loss.")

        detection_loss, detached_items = super().loss(predictions, batch)
        target = selective_degradation_target(
            batch["img"],
            batch["batch_idx"],
            batch["bboxes"],
            output_size=tuple(reconstruction.shape[-2:]),
        )
        reconstruction_loss = F.mse_loss(reconstruction.float(), target.float(), reduction="mean")
        batch_size = int(reconstruction.shape[0])
        detection_mean = detection_loss.sum() / batch_size
        log_variances = self.dre_head.dre_log_variances
        combined = (
            0.5 * torch.exp(-log_variances[0]) * detection_mean
            + 0.5 * log_variances[0]
            + 0.5 * torch.exp(-log_variances[1]) * reconstruction_loss
            + 0.5 * log_variances[1]
        )
        self.dre_head.last_reconstruction_loss = reconstruction_loss.detach()
        return combined * batch_size, detached_items
