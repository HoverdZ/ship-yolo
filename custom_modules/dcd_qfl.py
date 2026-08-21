"""IoU-supervised Quality Focal Loss for the unchanged DCD head.

This module changes only the classification supervision used by DCD. The DCD
spatial architecture, TaskAlignedAssigner, bbox loss, DFL loss, decoding, and
inference paths remain unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from custom_modules.dcd_head import DCDDetect


class DCDQFLDetect(DCDDetect):
    """Architecture-identical DCD marker used to select the QFL criterion."""


def aligned_iou_xyxy(
    box1: torch.Tensor,
    box2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return aligned axis-aligned IoU for matching ``[N, 4]`` xyxy boxes."""

    if box1.ndim != 2 or box1.shape[-1] != 4:
        raise ValueError(f"box1 must have shape [N, 4], got {tuple(box1.shape)}")
    if box2.shape != box1.shape:
        raise ValueError(
            "box2 must match box1 shape for aligned IoU, got "
            f"{tuple(box2.shape)} and {tuple(box1.shape)}"
        )

    left_top = torch.maximum(box1[:, :2], box2[:, :2])
    right_bottom = torch.minimum(box1[:, 2:], box2[:, 2:])
    intersection_wh = (right_bottom - left_top).clamp_min(0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

    box1_wh = (box1[:, 2:] - box1[:, :2]).clamp_min(0)
    box2_wh = (box2[:, 2:] - box2[:, :2]).clamp_min(0)
    area1 = box1_wh[:, 0] * box1_wh[:, 1]
    area2 = box2_wh[:, 0] * box2_wh[:, 1]
    union = (area1 + area2 - intersection).clamp_min(eps)
    return (intersection / union).clamp_(0, 1)


class QualityFocalLoss(nn.Module):
    """Standard quality focal classification loss with fixed beta=2."""

    beta = 2.0

    def forward(
        self,
        pred_logits: torch.Tensor,
        quality_targets: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return unreduced QFL for logits and same-shaped quality targets."""

        if pred_logits.shape != quality_targets.shape:
            raise ValueError(
                "QFL logits and targets must have identical shapes, got "
                f"{tuple(pred_logits.shape)} and {tuple(quality_targets.shape)}"
            )
        if pred_logits.device != quality_targets.device:
            raise ValueError("QFL logits and targets must be on the same device.")
        if pred_logits.dtype != quality_targets.dtype:
            raise ValueError("QFL logits and targets must use the same dtype.")

        pred_prob = pred_logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(
            pred_logits,
            quality_targets,
            reduction="none",
        )
        qfl = bce * (quality_targets - pred_prob).abs().pow(self.beta)
        if class_weights is not None:
            qfl *= class_weights
        return qfl


class DCDQFLDetectionLoss(v8DetectionLoss):
    """Native YOLO11 detection loss with only BCE classification changed to QFL."""

    def __init__(
        self,
        model: nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
    ) -> None:
        super().__init__(model=model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if not isinstance(model.model[-1], DCDQFLDetect):
            raise TypeError(
                "DCDQFLDetectionLoss requires DCDQFLDetect as the final model layer."
            )
        self.qfl = QualityFocalLoss()

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple:
        """Run native TAL/regression and supervise classification with detached IoU."""

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(
            preds["feats"], self.stride, 0.5
        )

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(
                preds["feats"][0].shape[2:],
                device=self.device,
                dtype=dtype,
            )
            * self.stride[0]
        )

        # Targets: unchanged from Ultralytics 8.4.92 v8DetectionLoss.
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes: unchanged native DFL decoding in grid coordinates.
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # TAL still exclusively selects positives and matches anchors to GT.
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = (
            self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
        )
        fg_mask = fg_mask.bool()

        # Preserve the native TAL score normalization for bbox and DFL only.
        target_scores_sum = max(target_scores.sum(), 1)

        if pred_scores.ndim != 3:
            raise ValueError(
                f"pred_scores must have shape [B, N, nc], got {tuple(pred_scores.shape)}"
            )
        if pred_bboxes.shape != (*pred_scores.shape[:2], 4):
            raise ValueError(
                "pred_bboxes must align with pred_scores anchors, got "
                f"{tuple(pred_bboxes.shape)} and {tuple(pred_scores.shape)}"
            )
        if target_bboxes.shape != pred_bboxes.shape:
            raise ValueError(
                "target_bboxes must align with pred_bboxes, got "
                f"{tuple(target_bboxes.shape)} and {tuple(pred_bboxes.shape)}"
            )
        if fg_mask.shape != pred_scores.shape[:2]:
            raise ValueError(
                "fg_mask must have shape [B, N], got "
                f"{tuple(fg_mask.shape)}"
            )

        # Classification quality targets: negatives remain zero; each positive
        # receives detached plain IoU at its TAL-matched class position.
        quality_targets = torch.zeros_like(pred_scores)
        if fg_mask.any():
            with torch.no_grad():
                pred_bboxes_px = pred_bboxes.detach() * stride_tensor
                positive_pred_boxes = pred_bboxes_px[fg_mask].float()
                positive_target_boxes = target_bboxes[fg_mask].detach().float()
                positive_iou = aligned_iou_xyxy(
                    positive_pred_boxes,
                    positive_target_boxes,
                ).to(device=pred_scores.device, dtype=dtype)

                positive_labels = target_labels[fg_mask].long().reshape(-1)
                if positive_labels.numel() != positive_iou.numel():
                    raise ValueError(
                        "Positive labels and IoU targets must have equal length, got "
                        f"{positive_labels.numel()} and {positive_iou.numel()}"
                    )
                if (
                    positive_labels.lt(0).any()
                    or positive_labels.ge(self.nc).any()
                ):
                    raise ValueError("TAL returned a positive class index outside [0, nc).")

                positive_quality = pred_scores.new_zeros(
                    (positive_labels.numel(), self.nc)
                )
                positive_quality.scatter_(
                    1,
                    positive_labels.unsqueeze(1),
                    positive_iou.unsqueeze(1),
                )
                quality_targets[fg_mask] = positive_quality

        qfl = self.qfl(pred_scores, quality_targets, self.class_weights)
        qfl_avg_factor = (
            fg_mask.sum().to(device=pred_scores.device, dtype=dtype).clamp_min(1.0)
        )
        loss[1] = qfl.sum() / qfl_avg_factor

        # Bbox and DFL are byte-for-byte equivalent in logic to the native path.
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


__all__ = [
    "DCDQFLDetect",
    "DCDQFLDetectionLoss",
    "QualityFocalLoss",
    "aligned_iou_xyxy",
]
