"""Original Anomaly-Aware Detection Head adapted to Ultralytics YOLO11.

The native YOLO11 regression and classification towers remain unchanged.
AADH adds one independent objectness branch to each of the P3/P4/P5 feature
maps and estimates objectness with the paper's exponential-background test.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou


class AADHFilteringBlock(nn.Module):
    """Two original 3x3 Conv-BN-ReLU filters producing eight channels."""

    def __init__(self, c_in: int, aa_channels: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, aa_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(aa_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(aa_channels, aa_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(aa_channels)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter one detection-scale feature map without changing its size."""

        return self.act2(self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x))))))


class _LogGammaSurvival(torch.autograd.Function):
    """Stable log of the regularized upper incomplete Gamma function.

    For arguments above 40, the forward expression follows the asymptotic
    approximation stated in the paper. The custom backward differentiates the
    exact and asymptotic expressions used by the corresponding forward region.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, shape: int) -> torch.Tensor:
        if x.dtype != torch.float32:
            raise TypeError("AADH statistical testing must enter Gamma evaluation in FP32.")
        shape_value = int(shape)
        if shape_value < 1:
            raise ValueError("Gamma shape must be a positive integer.")

        shape_tensor = x.new_tensor(float(shape_value))
        survival = torch.special.gammaincc(shape_tensor, x)
        exact_log = survival.clamp_min(torch.finfo(x.dtype).tiny).log()

        x_safe = x.clamp_min(torch.finfo(x.dtype).tiny)
        x_asymptotic = x.clamp_min(40.0)
        shape_minus_one = float(shape_value - 1)
        asymptotic_log = (
            -x_asymptotic
            + shape_minus_one * x_asymptotic.log()
            + torch.log1p(shape_minus_one / x_asymptotic)
            - math.lgamma(shape_value)
        )
        asymptotic_mask = x > 40.0
        log_survival = torch.where(asymptotic_mask, asymptotic_log, exact_log)

        ctx.shape = shape_value
        ctx.save_for_backward(x, log_survival, asymptotic_mask)
        return log_survival

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        x, log_survival, asymptotic_mask = ctx.saved_tensors
        shape_minus_one = float(ctx.shape - 1)
        x_safe = x.clamp_min(torch.finfo(x.dtype).tiny)
        x_asymptotic = x.clamp_min(40.0)

        exact_derivative = -torch.exp(
            shape_minus_one * x_safe.log()
            - x_safe
            - math.lgamma(ctx.shape)
            - log_survival
        )
        asymptotic_derivative = (
            -1.0
            + shape_minus_one / x_asymptotic
            - shape_minus_one
            / (x_asymptotic * (x_asymptotic + shape_minus_one))
        )
        derivative = torch.where(
            asymptotic_mask,
            asymptotic_derivative,
            exact_derivative,
        )
        derivative = torch.where(x > 0.0, derivative, torch.zeros_like(derivative))
        return grad_output * derivative, None


class AADHStatisticalTest(nn.Module):
    """Paper-defined mu2 exponential-background anomaly test."""

    def __init__(self, aa_channels: int = 8, alpha: float = 0.001) -> None:
        super().__init__()
        self.aa_channels = int(aa_channels)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one anomaly-derived objectness probability per spatial point."""

        if x.ndim != 4 or x.shape[1] != self.aa_channels:
            raise ValueError(
                "AADH latent features must have shape [B, C, H, W] with "
                f"C={self.aa_channels}, got {tuple(x.shape)}."
            )

        output_dtype = x.dtype
        x_stat = x.float()
        mean_activation = x_stat.mean(dim=(1, 2, 3), keepdim=True)
        batch_mean_activation = mean_activation.mean(dim=0).detach()
        if not hasattr(self, "lambda_ema"):
            self.register_buffer("lambda_ema", batch_mean_activation.clone())

        current_lambda = 1.0 / (mean_activation + 1e-7)
        if self.training:
            with torch.no_grad():
                self.lambda_ema.mul_(0.9).add_(batch_mean_activation, alpha=0.1)
            effective_lambda = current_lambda
        else:
            effective_lambda = (
                0.07 / (self.lambda_ema.float() + 1e-7)
                + 0.93 * current_lambda
            )

        scaled_activation = effective_lambda * x_stat
        mu2 = scaled_activation.sum(dim=1, keepdim=True).clamp_min(0.0)

        log_survival = _LogGammaSurvival.apply(mu2, self.aa_channels)
        significance = (-log_survival).clamp_min(0.0)
        objectness = 2.0 * torch.sigmoid(self.alpha * significance) - 1.0
        return objectness.to(dtype=output_dtype)


class AADHDetect(Detect):
    """Native YOLO11 Detect plus an original AADH objectness branch."""

    def __init__(
        self,
        nc: int = 80,
        aa_channels: int = 8,
        alpha: float = 0.001,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] | list[int] = (),
    ) -> None:
        if isinstance(nc, bool) or not isinstance(nc, int):
            raise TypeError("nc must be an int.")
        if isinstance(aa_channels, bool) or not isinstance(aa_channels, int):
            raise TypeError("aa_channels must be an int.")
        if aa_channels != 8:
            raise ValueError("Original AADH requires aa_channels=8.")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a real number.")
        alpha_value = float(alpha)
        if not math.isclose(alpha_value, 0.001, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Original AADH requires alpha=0.001.")
        if isinstance(reg_max, bool) or not isinstance(reg_max, int):
            raise TypeError("reg_max must be an int.")
        # Ultralytics 8.4.92 uses None as the parser sentinel when the YAML
        # omits end2end; it is the standard one-to-many path, equivalent to False.
        if end2end is None:
            end2end = False
        if not isinstance(end2end, bool):
            raise TypeError("end2end must be a bool.")
        if end2end:
            raise NotImplementedError(
                "AADHDetect supports only the standard YOLO11 one-to-many Detect path."
            )
        if not isinstance(ch, (tuple, list)) or len(ch) != 3:
            raise ValueError("Original AADH requires exactly the P3/P4/P5 input features.")
        if any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in ch):
            raise TypeError("ch must contain three positive integer channel counts.")

        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=tuple(ch))
        self.aa_channels = aa_channels
        self.alpha = alpha_value
        self.aadh_filters = nn.ModuleList(
            AADHFilteringBlock(c_in=c, aa_channels=aa_channels) for c in ch
        )
        self.aadh_test = AADHStatisticalTest(
            aa_channels=aa_channels,
            alpha=alpha_value,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, Any]:
        """Run native box/class towers and append P3/P4/P5 objectness maps."""

        predictions = super().forward_head(x, box_head, cls_head)
        if predictions:
            if len(x) != self.nl or len(self.aadh_filters) != self.nl:
                raise ValueError("AADHDetect expects one objectness branch per detection scale.")
            predictions["objectness"] = [
                self.aadh_test(aadh_filter(feature))
                for aadh_filter, feature in zip(self.aadh_filters, x)
            ]
        return predictions

    def _inference(self, predictions: dict[str, Any]) -> torch.Tensor:
        """Decode native boxes and rank classes by class probability times objectness."""

        objectness_maps = predictions.get("objectness")
        if not isinstance(objectness_maps, list) or len(objectness_maps) != self.nl:
            raise ValueError("AADH inference requires P3/P4/P5 objectness maps.")

        batch_size = predictions["scores"].shape[0]
        objectness = torch.cat(
            [level.reshape(batch_size, 1, -1) for level in objectness_maps],
            dim=2,
        )
        class_probabilities = predictions["scores"].sigmoid()
        if objectness.shape[2] != class_probabilities.shape[2]:
            raise ValueError("AADH objectness points must align with native class predictions.")
        objectness = objectness.to(dtype=class_probabilities.dtype)
        final_scores = class_probabilities * objectness
        return torch.cat((self._get_decode_boxes(predictions), final_scores), dim=1)


class AADHDetectionLoss(v8DetectionLoss):
    """Native YOLO11 loss plus original CIoU-supervised AADH MSE objectness."""

    objectness_balance = (4.0, 1.0, 0.4)
    objectness_gain = 0.7

    def __init__(
        self,
        model: nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
    ) -> None:
        super().__init__(model=model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if not isinstance(model.model[-1], AADHDetect):
            raise TypeError("AADHDetectionLoss requires AADHDetect as the final model layer.")
        for gain_name in ("box", "cls", "dfl"):
            gain = getattr(self.hyp, gain_name)
            if isinstance(gain, bool) or not isinstance(gain, (int, float)):
                raise TypeError(f"YOLO11 {gain_name} gain must be a real number.")

    def _objectness_loss(
        self,
        predictions: dict[str, Any],
        fg_mask: torch.Tensor,
        target_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Build detached CIoU targets from the existing TAL match and apply MSE."""

        objectness_maps = predictions.get("objectness")
        if not isinstance(objectness_maps, list) or len(objectness_maps) != 3:
            raise ValueError("AADH loss requires three P3/P4/P5 objectness maps.")

        pred_scores = predictions["scores"].permute(0, 2, 1).contiguous()
        if fg_mask.shape != pred_scores.shape[:2]:
            raise ValueError(
                "fg_mask must align with native YOLO11 prediction points, got "
                f"{tuple(fg_mask.shape)} and {tuple(pred_scores.shape[:2])}."
            )
        target_objectness = pred_scores.new_zeros(pred_scores.shape[:2])

        if fg_mask.any():
            with torch.no_grad():
                pred_distribution = (
                    predictions["boxes"].detach().permute(0, 2, 1).contiguous()
                )
                pred_bboxes = self.bbox_decode(anchor_points, pred_distribution)
                pred_bboxes_px = pred_bboxes * stride_tensor
                positive_ciou = bbox_iou(
                    pred_bboxes_px[fg_mask].float(),
                    target_bboxes[fg_mask].detach().float(),
                    xywh=False,
                    CIoU=True,
                ).detach().reshape(-1).clamp_(0.0, 1.0)
                positive_ciou = positive_ciou.to(
                    device=pred_scores.device,
                    dtype=pred_scores.dtype,
                )
            target_objectness[fg_mask] = positive_ciou

        level_sizes: list[int] = []
        batch_size = pred_scores.shape[0]
        for level in objectness_maps:
            if level.ndim != 4 or level.shape[:2] != (batch_size, 1):
                raise ValueError(
                    "Each AADH objectness map must have shape [B, 1, H, W], got "
                    f"{tuple(level.shape)}."
                )
            level_sizes.append(level.shape[2] * level.shape[3])
        if sum(level_sizes) != pred_scores.shape[1]:
            raise ValueError("AADH objectness maps do not match native prediction count.")

        objectness_loss = pred_scores.new_zeros((), dtype=torch.float32)
        offset = 0
        for level, level_size, balance in zip(
            objectness_maps,
            level_sizes,
            self.objectness_balance,
        ):
            level_target = target_objectness[:, offset : offset + level_size]
            level_target = level_target.reshape_as(level)
            objectness_loss = objectness_loss + float(balance) * F.mse_loss(
                level.float(),
                level_target.float(),
            )
            offset += level_size
        return objectness_loss

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple:
        """Use native TAL/base losses once, then add the original AADH MSE term."""

        assigned, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned
        fg_mask = fg_mask.bool()
        objectness_loss = self._objectness_loss(
            preds,
            fg_mask,
            target_bboxes,
            anchor_points,
            stride_tensor,
        )

        # Ultralytics 8.4.92 exposes three trainer loss slots. The AADH term is
        # added to the existing cls reporting slot without changing cls BCE.
        loss[1] = loss[1] + float(self.objectness_gain) * objectness_loss
        assigned = (
            fg_mask,
            target_gt_idx,
            target_bboxes,
            anchor_points,
            stride_tensor,
        )
        return assigned, loss, loss.detach()


__all__ = [
    "AADHDetect",
    "AADHDetectionLoss",
    "AADHFilteringBlock",
    "AADHStatisticalTest",
]
