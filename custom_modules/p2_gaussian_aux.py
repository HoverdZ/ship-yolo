"""Training-only P2 Gaussian supervision for YOLO11 detection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class P2GaussianAuxDetect(Detect):
    """Keep Detect(P3,P4,P5) and add a P2 heatmap only during training.

    In evaluation/export mode the auxiliary convolution is not executed, so
    inference outputs and inference compute remain those of the native
    three-scale Detect head.
    """

    def __init__(
        self,
        nc: int = 80,
        aux_weight: float = 0.25,
        sigma_scale: float = 0.25,
        min_sigma: float = 1.0,
        max_sigma: float = 3.0,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        if len(ch) != 4:
            raise ValueError(
                "P2GaussianAuxDetect requires channels for [P2, P3, P4, P5], "
                f"got {ch}."
            )
        if aux_weight <= 0:
            raise ValueError("aux_weight must be positive.")
        if sigma_scale <= 0 or min_sigma <= 0 or max_sigma < min_sigma:
            raise ValueError("Invalid Gaussian sigma configuration.")
        self.p2_channels = int(ch[0])
        self.aux_weight = float(aux_weight)
        self.sigma_scale = float(sigma_scale)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=tuple(ch[1:]))
        self.p2_aux = nn.Conv2d(self.p2_channels, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.p2_aux.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.p2_aux.bias, -4.59511985013459)  # prior p=0.01

    def forward(self, x: list[torch.Tensor]) -> Any:
        """Return native Detect output plus auxiliary logits only in training."""

        if len(x) != 4:
            raise ValueError(f"Expected [P2,P3,P4,P5], got {len(x)} tensors.")
        if self.training:
            aux_logits = self.p2_aux(x[0])
            predictions = super().forward(x[1:])
            if not isinstance(predictions, dict):
                raise RuntimeError("Native Detect training output must be a dictionary.")
            predictions["p2_aux_logits"] = aux_logits
            return predictions
        return super().forward(x[1:])


def gaussian_heatmap_targets(
    batch: dict[str, torch.Tensor],
    shape: tuple[int, int, int, int],
    *,
    sigma_scale: float,
    min_sigma: float,
    max_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rasterize normalized YOLO boxes as max-composed P2 Gaussian heatmaps."""

    batch_size, channels, height, width = shape
    if channels != 1:
        raise ValueError(f"Auxiliary logits must have one channel, got {channels}.")
    heatmap = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    if batch["bboxes"].numel() == 0:
        return heatmap

    boxes = batch["bboxes"].to(device=device, dtype=dtype)
    indices = batch["batch_idx"].view(-1).to(device=device, dtype=torch.long)
    yy = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, width)

    for box, image_index in zip(boxes, indices):
        cx = torch.clamp(torch.round(box[0] * width - 0.5), 0, width - 1)
        cy = torch.clamp(torch.round(box[1] * height - 0.5), 0, height - 1)
        box_w = torch.clamp(box[2] * width, min=1.0)
        box_h = torch.clamp(box[3] * height, min=1.0)
        sigma = torch.clamp(
            sigma_scale * torch.sqrt(box_w * box_h),
            min=min_sigma,
            max=max_sigma,
        )
        gaussian = torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2 * sigma.square()))
        heatmap[image_index, 0] = torch.maximum(heatmap[image_index, 0], gaussian)
    return heatmap


def dense_gaussian_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Balanced dense focal BCE for soft Gaussian targets, evaluated in FP32.

    Keeping the logits in FP16 is unsafe: ``1 - 1e-6`` rounds to ``1.0`` and
    saturated sigmoid values make ``log(1 - p)`` infinite. FP32 log-sigmoid
    identities remain finite for all finite logits while preserving AMP for
    the rest of the model.
    """

    if logits.shape != target.shape:
        raise ValueError(f"Logit/target shape mismatch: {logits.shape} != {target.shape}.")
    logits_fp32 = logits.float()
    target_fp32 = target.float()
    probability = logits_fp32.sigmoid()
    positive_mass = target_fp32.sum().clamp_min(1.0)
    positive = -(
        target_fp32
        * (1 - probability).square()
        * F.logsigmoid(logits_fp32)
    ).sum() / positive_mass
    negative_weight = (1 - target_fp32).pow(4)
    negative = -(
        negative_weight
        * probability.square()
        * F.logsigmoid(-logits_fp32)
    ).mean()
    return positive + negative


class P2GaussianAuxLoss(v8DetectionLoss):
    """Native YOLO loss plus a weighted training-only P2 heatmap loss."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        head = model.model[-1]
        if not isinstance(head, P2GaussianAuxDetect):
            raise TypeError("P2GaussianAuxLoss requires P2GaussianAuxDetect.")
        self.aux_weight = head.aux_weight
        self.sigma_scale = head.sigma_scale
        self.min_sigma = head.min_sigma
        self.max_sigma = head.max_sigma

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute native detection losses and append the weighted auxiliary loss."""

        detection_total, detection_items = super().loss(preds, batch)
        logits = preds.get("p2_aux_logits")
        if logits is None:
            auxiliary = detection_total.new_zeros(())
        else:
            target = gaussian_heatmap_targets(
                batch,
                tuple(logits.shape),
                sigma_scale=self.sigma_scale,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                device=logits.device,
                dtype=logits.dtype,
            )
            auxiliary = dense_gaussian_focal_loss(logits, target)
        weighted_auxiliary = self.aux_weight * auxiliary
        batch_size = preds["boxes"].shape[0]
        total = torch.cat(
            (detection_total, (weighted_auxiliary * batch_size).view(1))
        )
        items = torch.cat((detection_items, weighted_auxiliary.detach().view(1)))
        return total, items


class P2GaussianDetectionModel(DetectionModel):
    """DetectionModel whose criterion understands the auxiliary P2 output."""

    def init_criterion(self) -> P2GaussianAuxLoss:
        return P2GaussianAuxLoss(self)
