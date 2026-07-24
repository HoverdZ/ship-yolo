"""Scale-adaptive Gaussian task alignment for tiny-object YOLO training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist


def normalized_wasserstein_similarity(
    first_xyxy: torch.Tensor,
    second_xyxy: torch.Tensor,
    constant: float = 12.8,
) -> torch.Tensor:
    """Return NWD similarity for paired horizontal boxes in pixel units."""

    if first_xyxy.shape != second_xyxy.shape or first_xyxy.shape[-1] != 4:
        raise ValueError(
            f"NWD requires paired (...,4) boxes, got {first_xyxy.shape} and {second_xyxy.shape}."
        )
    if constant <= 0:
        raise ValueError("NWD normalization constant must be positive.")
    first_center = (first_xyxy[..., :2] + first_xyxy[..., 2:]) * 0.5
    second_center = (second_xyxy[..., :2] + second_xyxy[..., 2:]) * 0.5
    first_wh = (first_xyxy[..., 2:] - first_xyxy[..., :2]).clamp_min(0)
    second_wh = (second_xyxy[..., 2:] - second_xyxy[..., :2]).clamp_min(0)
    distance_squared = (
        (first_center - second_center).square().sum(dim=-1)
        + 0.25 * (first_wh - second_wh).square().sum(dim=-1)
    )
    return torch.exp(-torch.sqrt(distance_squared.clamp_min(0) + 1e-9) / constant)


def iou_blend_weight(
    target_xyxy: torch.Tensor,
    threshold: float = 32.0,
    temperature: float = 6.0,
) -> torch.Tensor:
    """Smoothly shift from NWD for tiny GT boxes to IoU for larger boxes."""

    if threshold <= 0 or temperature <= 0:
        raise ValueError("threshold and temperature must be positive.")
    target_wh = (target_xyxy[..., 2:] - target_xyxy[..., :2]).clamp_min(0)
    target_scale = torch.sqrt(target_wh[..., 0] * target_wh[..., 1] + 1e-9)
    return torch.sigmoid((target_scale - threshold) / temperature)


def scale_adaptive_quality(
    target_xyxy: torch.Tensor,
    predicted_xyxy: torch.Tensor,
    *,
    threshold: float = 32.0,
    temperature: float = 6.0,
    nwd_constant: float = 12.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return blended quality, CIoU, and NWD for paired pixel-space boxes."""

    ciou = bbox_iou(target_xyxy, predicted_xyxy, xywh=False, CIoU=True)
    if ciou.ndim == target_xyxy.ndim:
        ciou = ciou.squeeze(-1)
    ciou = ciou.clamp(min=0.0, max=1.0)
    nwd = normalized_wasserstein_similarity(
        target_xyxy, predicted_xyxy, constant=nwd_constant
    )
    blend = iou_blend_weight(
        target_xyxy, threshold=threshold, temperature=temperature
    )
    return blend * ciou + (1.0 - blend) * nwd, ciou, nwd


class ScaleAdaptiveGaussianAssigner(TaskAlignedAssigner):
    """TaskAlignedAssigner using size-adaptive IoU/NWD localization quality."""

    def __init__(
        self,
        *args,
        threshold: float = 32.0,
        temperature: float = 6.0,
        nwd_constant: float = 12.8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.nwd_constant = float(nwd_constant)

    def iou_calculation(
        self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor
    ) -> torch.Tensor:
        quality, _, _ = scale_adaptive_quality(
            gt_bboxes,
            pd_bboxes,
            threshold=self.threshold,
            temperature=self.temperature,
            nwd_constant=self.nwd_constant,
        )
        return quality


class ScaleAdaptiveGaussianBboxLoss(BboxLoss):
    """Blend CIoU and NWD for box loss while keeping native DFL unchanged."""

    def __init__(
        self,
        reg_max: int = 16,
        *,
        threshold: float = 32.0,
        temperature: float = 6.0,
        nwd_constant: float = 12.8,
    ) -> None:
        super().__init__(reg_max)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.nwd_constant = float(nwd_constant)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        positive_stride = stride.unsqueeze(0).expand(
            fg_mask.shape[0], -1, -1
        )[fg_mask]
        predicted_pixels = pred_bboxes[fg_mask] * positive_stride
        target_pixels = target_bboxes[fg_mask] * positive_stride
        quality, _, _ = scale_adaptive_quality(
            target_pixels,
            predicted_pixels,
            threshold=self.threshold,
            temperature=self.temperature,
            nwd_constant=self.nwd_constant,
        )
        loss_box = ((1.0 - quality).unsqueeze(-1) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(
                    pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none"
                ).mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_box, loss_dfl


class ScaleAdaptiveGaussianDetectionLoss(v8DetectionLoss):
    """Native YOLO11 loss with matched Gaussian assignment and box quality."""

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 32.0,
        temperature: float = 6.0,
        nwd_constant: float = 12.8,
    ) -> None:
        super().__init__(model)
        self.assigner = ScaleAdaptiveGaussianAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            threshold=threshold,
            temperature=temperature,
            nwd_constant=nwd_constant,
        )
        self.bbox_loss = ScaleAdaptiveGaussianBboxLoss(
            self.reg_max,
            threshold=threshold,
            temperature=temperature,
            nwd_constant=nwd_constant,
        ).to(self.device)


class SGTADetectionModel(DetectionModel):
    """DetectionModel whose criterion applies SGTA only during training."""

    def init_criterion(self) -> ScaleAdaptiveGaussianDetectionLoss:
        return ScaleAdaptiveGaussianDetectionLoss(self)
