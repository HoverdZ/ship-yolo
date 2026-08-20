"""Relative-Uncertainty Quality head for Ultralytics YOLO11.

RUQ reuses the native DFL regression logits. It estimates localization quality
from distribution entropy, standard deviation, and scale-normalized boundary
uncertainty, without replacing the native regression or classification towers.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss


def aligned_box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Return aligned IoU for two ``xyxy`` tensors with the same shape."""

    intersection_lt = torch.maximum(box1[..., :2], box2[..., :2])
    intersection_rb = torch.minimum(box1[..., 2:], box2[..., 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    box1_wh = (box1[..., 2:] - box1[..., :2]).clamp_min(0)
    box2_wh = (box2[..., 2:] - box2[..., :2]).clamp_min(0)
    area1 = box1_wh[..., 0] * box1_wh[..., 1]
    area2 = box2_wh[..., 0] * box2_wh[..., 1]
    return intersection / (area1 + area2 - intersection + eps)


class RUQDetect(Detect):
    """YOLO11 Detect head with DFL-derived relative-uncertainty calibration.

    The original ``cv2`` regression, ``cv3`` classification, DFL decoder, and
    TaskAlignedAssigner path are preserved. A shared pointwise predictor reads
    ten statistics from the four DFL distributions:

    - four normalized entropies;
    - four normalized standard deviations;
    - horizontal and vertical relative uncertainty.

    Relative uncertainty is computed in grid units, where stride cancels from
    numerator and denominator. It is therefore equivalent to pixel-space
    uncertainty divided by the predicted box width or height.
    """

    statistic_channels = 10

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
        quality_hidden: int = 16,
        quality_gain: float = 0.5,
        quality_negative_weight: float = 0.05,
        quality_focal_gamma: float = 2.0,
        detach_distribution: bool = True,
        relative_uncertainty_cap: float = 4.0,
        quality_prior: float = 0.90,
    ) -> None:
        if end2end:
            raise ValueError("RUQDetect supports the standard YOLO11 Detect path only; end2end must be false.")
        if reg_max <= 1:
            raise ValueError("RUQDetect requires DFL with reg_max > 1.")
        if quality_hidden < 1:
            raise ValueError("quality_hidden must be positive.")
        if quality_gain < 0.0 or quality_negative_weight < 0.0 or quality_focal_gamma < 0.0:
            raise ValueError("RUQ loss weights and focal gamma must be non-negative.")
        if relative_uncertainty_cap <= 0.0:
            raise ValueError("relative_uncertainty_cap must be positive.")
        if not 0.0 < quality_prior < 1.0:
            raise ValueError("quality_prior must be strictly between 0 and 1.")

        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.quality_gain = float(quality_gain)
        self.quality_negative_weight = float(quality_negative_weight)
        self.quality_focal_gamma = float(quality_focal_gamma)
        self.detach_distribution = bool(detach_distribution)
        self.relative_uncertainty_cap = float(relative_uncertainty_cap)
        self.quality_prior = float(quality_prior)

        self.quality_predictor = nn.Sequential(
            nn.Conv1d(self.statistic_channels, quality_hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(quality_hidden, 1, kernel_size=1, bias=True),
        )
        self.register_buffer(
            "_ruq_bins",
            torch.arange(self.reg_max, dtype=torch.float32),
            persistent=False,
        )
        self._initialize_quality_predictor()

    def _initialize_quality_predictor(self) -> None:
        """Start close to identity calibration while retaining trainable gradients."""

        final = self.quality_predictor[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(self.quality_prior / (1.0 - self.quality_prior)))

    def bias_init(self) -> None:
        """Initialize native Detect biases and restore the explicit RUQ prior."""

        super().bias_init()
        self._initialize_quality_predictor()

    def distribution_statistics(self, box_logits: torch.Tensor) -> torch.Tensor:
        """Convert ``4 * reg_max`` DFL logits into ten uncertainty statistics."""

        batch, channels, anchors = box_logits.shape
        expected_channels = 4 * self.reg_max
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} DFL channels, received {channels}.")

        # Float32 statistics avoid half-precision log/variance instability under AMP.
        logits = box_logits.view(batch, 4, self.reg_max, anchors).float()
        probabilities = logits.softmax(dim=2)
        bins = self._ruq_bins.to(device=probabilities.device).view(1, 1, self.reg_max, 1)

        means = (probabilities * bins).sum(dim=2)
        variances = (probabilities * (bins - means.unsqueeze(2)).square()).sum(dim=2)
        standard_deviations = (variances + 1e-9).sqrt()
        normalized_std = standard_deviations / float(self.reg_max - 1)

        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=2)
        normalized_entropy = entropy / math.log(self.reg_max)

        predicted_width = (means[:, 0] + means[:, 2]).clamp_min(1e-3)
        predicted_height = (means[:, 1] + means[:, 3]).clamp_min(1e-3)
        relative_x = (standard_deviations[:, 0] + standard_deviations[:, 2]) / predicted_width
        relative_y = (standard_deviations[:, 1] + standard_deviations[:, 3]) / predicted_height
        relative = torch.stack((relative_x, relative_y), dim=1).clamp(
            min=0.0,
            max=self.relative_uncertainty_cap,
        )

        statistics = torch.cat((normalized_entropy, normalized_std, relative), dim=1)
        if self.detach_distribution:
            statistics = statistics.detach()
        return statistics.to(dtype=box_logits.dtype)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run native Detect towers and append one localization-quality logit."""

        predictions = super().forward_head(x, box_head, cls_head)
        if predictions:
            statistics = self.distribution_statistics(predictions["boxes"])
            predictions["quality"] = self.quality_predictor(statistics)
            predictions["ruq_statistics"] = statistics
        return predictions

    def _inference(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode native boxes and calibrate class scores before NMS ranking."""

        boxes = self._get_decode_boxes(predictions)
        calibrated_scores = predictions["scores"].sigmoid() * predictions["quality"].sigmoid()
        return torch.cat((boxes, calibrated_scores), dim=1)


class RUQDetectionLoss(v8DetectionLoss):
    """Native YOLO11 loss plus IoU-supervised RUQ calibration.

    The quality term is added to the existing classification loss slot. This
    deliberately keeps Ultralytics' three-value ``box/cls/dfl`` trainer and CSV
    interfaces unchanged.
    """

    def __init__(self, model: nn.Module, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        super().__init__(model=model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        head = model.model[-1]
        if not isinstance(head, RUQDetect):
            raise TypeError("RUQDetectionLoss requires RUQDetect as the final model layer.")
        self.head = head
        self.last_quality_loss = torch.tensor(0.0, device=self.device)
        self.last_positive_quality = torch.tensor(0.0, device=self.device)

    def _quality_loss(
        self,
        predictions: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        target_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise quality with detached aligned IoU and focal-weighted negatives."""

        quality_logits = predictions["quality"].squeeze(1)
        pred_distribution = predictions["boxes"].permute(0, 2, 1).contiguous()
        pred_bboxes = self.bbox_decode(anchor_points, pred_distribution) * stride_tensor

        quality_target = torch.zeros_like(quality_logits)
        if fg_mask.any():
            positive_iou = aligned_box_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask]).detach().clamp_(0.0, 1.0)
            quality_target[fg_mask] = positive_iou
            self.last_positive_quality = positive_iou.mean().detach()
        else:
            self.last_positive_quality = quality_logits.new_zeros(())

        element_loss = F.binary_cross_entropy_with_logits(
            quality_logits,
            quality_target,
            reduction="none",
        )
        probability = quality_logits.sigmoid()
        focal_weight = (quality_target - probability).abs().pow(self.head.quality_focal_gamma)
        weighted_loss = element_loss * focal_weight

        # Normalize foreground and background independently. The number of
        # background locations grows quadratically with input resolution, so
        # normalizing everything by foreground count would make RUQ dominate
        # the native YOLO losses on sparse tiny-object datasets.
        if fg_mask.any():
            positive_loss = weighted_loss[fg_mask].mean()
        else:
            positive_loss = quality_logits.sum() * 0.0
        if self.head.quality_negative_weight > 0.0 and (~fg_mask).any():
            negative_loss = weighted_loss[~fg_mask].mean()
        else:
            negative_loss = quality_logits.sum() * 0.0
        return positive_loss + self.head.quality_negative_weight * negative_loss

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple:
        """Preserve TAL/base losses and add RUQ loss without changing log shape."""

        assigned, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, _, target_bboxes, anchor_points, stride_tensor = assigned
        quality_loss = self._quality_loss(
            preds,
            fg_mask,
            target_bboxes,
            anchor_points,
            stride_tensor,
        )
        self.last_quality_loss = quality_loss.detach()
        self.head.last_quality_loss = self.last_quality_loss
        self.head.last_positive_quality = self.last_positive_quality
        loss[1] = loss[1] + self.head.quality_gain * quality_loss
        return assigned, loss, loss.detach()


__all__ = ["RUQDetect", "RUQDetectionLoss", "aligned_box_iou"]
