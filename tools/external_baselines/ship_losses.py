"""Bounding-box losses and foreground trainers for paper reproductions.

Training still runs through the official Ultralytics ``YOLO.train`` API in the
current process.  The trainer subclasses only replace the detector criterion;
they do not launch subprocesses or alter the data/training loop.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.tal import bbox2dist


def _iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Aligned IoU for two equally sized ``xyxy`` tensors."""

    intersection_wh = (
        torch.minimum(box1[:, 2:], box2[:, 2:])
        - torch.maximum(box1[:, :2], box2[:, :2])
    ).clamp(min=0)
    intersection = intersection_wh.prod(dim=1, keepdim=True)
    area1 = (box1[:, 2:] - box1[:, :2]).clamp(min=0).prod(dim=1, keepdim=True)
    area2 = (box2[:, 2:] - box2[:, :2]).clamp(min=0).prod(dim=1, keepdim=True)
    return intersection / (area1 + area2 - intersection + eps)


def _scaled_xyxy(boxes: torch.Tensor, ratio: float) -> torch.Tensor:
    """Create center-aligned auxiliary boxes for Inner-IoU."""

    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    half_size = (boxes[:, 2:] - boxes[:, :2]) * (0.5 * ratio)
    return torch.cat((center - half_size, center + half_size), dim=1)


class _PaperBboxLoss(BboxLoss):
    """Shared DFL implementation with a paper-specific IoU term."""

    def _iou_loss(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

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
        raw_iou_loss = self._iou_loss(
            pred_bboxes,
            target_bboxes,
            fg_mask,
            imgsz,
            stride,
        )
        loss_iou = (raw_iou_loss * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.dfl_loss.reg_max - 1,
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            normalized_pred = pred_dist * stride
            normalized_pred[..., 0::2] /= imgsz[1]
            normalized_pred[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(
                    normalized_pred[fg_mask],
                    target_ltrb[fg_mask],
                    reduction="none",
                ).mean(-1, keepdim=True)
                * weight
            ).sum() / target_scores_sum
        return loss_iou, loss_dfl


class WiseIoUv3BboxLoss(_PaperBboxLoss):
    """WIoU-v3 with the published dynamic non-monotonic focusing rule."""

    def __init__(
        self,
        reg_max: int = 16,
        alpha: float = 1.7,
        delta: float = 2.7,
    ) -> None:
        super().__init__(reg_max)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.register_buffer("iou_loss_mean", torch.tensor(1.0))
        self.ema_momentum = 1.0 - 0.5 ** (1.0 / 7000.0)

    def _iou_loss(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        del imgsz, stride
        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]
        iou = _iou_xyxy(pred, target)
        base_loss = 1.0 - iou

        pred_center = (pred[:, :2] + pred[:, 2:]) * 0.5
        target_center = (target[:, :2] + target[:, 2:]) * 0.5
        center_distance = (pred_center - target_center).square().sum(1, keepdim=True)
        enclosing_lt = torch.minimum(pred[:, :2], target[:, :2])
        enclosing_rb = torch.maximum(pred[:, 2:], target[:, 2:])
        enclosing_diagonal = (enclosing_rb - enclosing_lt).square().sum(1, keepdim=True)
        distance_attention = torch.exp(
            center_distance / (enclosing_diagonal.detach() + 1e-7)
        )

        if self.training:
            with torch.no_grad():
                batch_mean = base_loss.detach().mean()
                self.iou_loss_mean.mul_(1.0 - self.ema_momentum).add_(
                    batch_mean * self.ema_momentum
                )
        beta = base_loss.detach() / self.iou_loss_mean.clamp(min=1e-4)
        focus = beta / (
            self.delta * torch.pow(self.alpha, beta - self.delta) + 1e-7
        )
        return focus * distance_attention * base_loss


class InnerMPDIoUBboxLoss(_PaperBboxLoss):
    """Inner-MPDIoU using auxiliary boxes and two-corner distance penalties."""

    def __init__(self, reg_max: int = 16, ratio: float = 0.7) -> None:
        super().__init__(reg_max)
        if not 0.0 < ratio <= 2.0:
            raise ValueError(f"Inner-IoU ratio must be in (0, 2], got {ratio}.")
        self.ratio = float(ratio)

    def _iou_loss(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred_bboxes[fg_mask]
        target = target_bboxes[fg_mask]
        inner_iou = _iou_xyxy(
            _scaled_xyxy(pred, self.ratio),
            _scaled_xyxy(target, self.ratio),
        )
        top_left_distance = (pred[:, :2] - target[:, :2]).square().sum(1, keepdim=True)
        bottom_right_distance = (pred[:, 2:] - target[:, 2:]).square().sum(1, keepdim=True)

        stride_per_anchor = stride.reshape(1, -1, 1).expand(
            fg_mask.shape[0], -1, -1
        )[fg_mask]
        feature_height = imgsz[0] / stride_per_anchor
        feature_width = imgsz[1] / stride_per_anchor
        normalizer = feature_height.square() + feature_width.square() + 1e-7
        return (
            1.0
            - inner_iou
            + top_left_distance / normalizer
            + bottom_right_distance / normalizer
        )


class WiseIoUv3DetectionLoss(v8DetectionLoss):
    """Ultralytics detection loss with WIoU-v3 regression."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.bbox_loss = WiseIoUv3BboxLoss(self.reg_max).to(self.device)


class InnerMPDIoUDetectionLoss(v8DetectionLoss):
    """Ultralytics detection loss with Inner-MPDIoU regression."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        ratio = float(model.yaml.get("inner_iou_ratio", 0.7))
        self.bbox_loss = InnerMPDIoUBboxLoss(self.reg_max, ratio=ratio).to(self.device)


class WiseIoUv3DetectionModel(DetectionModel):
    """Detection model whose criterion is WIoU-v3."""

    def init_criterion(self):
        return WiseIoUv3DetectionLoss(self)


class InnerMPDIoUDetectionModel(DetectionModel):
    """Detection model whose criterion is Inner-MPDIoU."""

    def init_criterion(self):
        return InnerMPDIoUDetectionLoss(self)


class _PaperDetectionTrainer(DetectionTrainer):
    """Base trainer that swaps only the DetectionModel criterion class."""

    model_class: type[DetectionModel]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = self.set_model_names_for_load(
            self.model_class(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose,
            )
        )
        if weights:
            model.load(weights)
        return model


class WiseIoUv3Trainer(_PaperDetectionTrainer):
    """Foreground Ultralytics trainer for APFAN and SHIP-YOLO."""

    model_class = WiseIoUv3DetectionModel


class InnerMPDIoUTrainer(_PaperDetectionTrainer):
    """Foreground Ultralytics trainer for PMF-YOLOv8."""

    model_class = InnerMPDIoUDetectionModel


TRAINERS = {
    "wise_iou_v3": WiseIoUv3Trainer,
    "inner_mpdiou": InnerMPDIoUTrainer,
}


def get_loss_trainer(loss_name: str | None):
    """Resolve an optional custom trainer without hiding the loss selection."""

    if loss_name in {None, "ciou"}:
        return None
    try:
        return TRAINERS[loss_name]
    except KeyError as error:
        raise KeyError(f"Unsupported paper reproduction loss: {loss_name!r}.") from error
