"""Focal CIoU loss isolated from the DPCSANet method for YOLO11."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class FocalCIoUBboxLoss(BboxLoss):
    """Replace only YOLO11's CIoU regression term with paper-defined Focal CIoU."""

    def __init__(self, reg_max: int = 16, gamma: float = 0.5) -> None:
        super().__init__(reg_max)
        self.gamma = float(gamma)
        if self.gamma <= 0.0:
            raise ValueError(f"Focal CIoU gamma must be positive, got {gamma}.")

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
        foreground_pred = pred_bboxes[fg_mask]
        foreground_target = target_bboxes[fg_mask]
        iou = bbox_iou(foreground_pred, foreground_target, xywh=False)
        ciou = bbox_iou(foreground_pred, foreground_target, xywh=False, CIoU=True)
        focal_weight = (1.0 - iou).clamp_min(0.0).pow(self.gamma)
        loss_iou = (focal_weight * (1.0 - ciou) * weight).sum() / target_scores_sum

        # Native YOLO11 DFL path is intentionally unchanged.
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
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
                F.l1_loss(normalized_pred[fg_mask], target_ltrb[fg_mask], reduction="none").mean(
                    -1, keepdim=True
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


class FocalCIoUDetect(Detect):
    """Architecture-identical Detect marker that selects the Focal CIoU criterion."""

    def __init__(
        self,
        nc: int = 80,
        gamma: float = 0.5,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.focal_ciou_gamma = float(gamma)


class FocalCIoUDetectionLoss(v8DetectionLoss):
    """YOLO11 detection loss with only the bbox IoU criterion exchanged."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        head = model.model[-1]
        gamma = getattr(head, "focal_ciou_gamma", 0.5)
        self.bbox_loss = FocalCIoUBboxLoss(self.reg_max, gamma=gamma).to(self.device)
