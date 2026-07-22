"""FaPN selection/alignment helpers that preserve YOLO11's native fusion.

The selection and offset-controller ideas follow the official ICCV 2021 FaPN
implementation (EMI-Group/FaPN, ``detectron2/modeling/backbone/fan.py``).
Unlike the original FaPN neck, these modules do not replace YOLO's nearest
upsampling, concatenation, top-down C3k2 blocks, PAN, or Detect head.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d


def _xavier_init(module: nn.Conv2d) -> None:
    """Use the same auditable Xavier family as the repository FaPN port."""

    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class FaPNFeatureSelectionKeep(nn.Module):
    """Select a shallow feature without changing its channels or resolution."""

    def __init__(self, channels: int, gamma_init: float = 0.1) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.in_channels = channels
        self.out_channels = channels
        self.conv_attention = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma_s = nn.Parameter(torch.tensor(float(gamma_init)))
        _xavier_init(self.conv_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x + gamma_s * x * sigmoid(conv1x1(GAP(x)))``."""

        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"FaPNFeatureSelectionKeep expects [B,{self.in_channels},H,W], "
                f"got {tuple(x.shape)}."
            )
        attention = torch.sigmoid(self.conv_attention(F.adaptive_avg_pool2d(x, 1)))
        return x + self.gamma_s * (x * attention)


class FaPNDepthwiseModulatedDeformConv2d(nn.Module):
    """Depthwise Torchvision DCNv2 with an independent offset controller."""

    kernel_size = 3

    def __init__(
        self,
        high_channels: int,
        controller_channels: int = 64,
        deformable_groups: int = 8,
    ) -> None:
        super().__init__()
        if high_channels <= 0 or controller_channels <= 0:
            raise ValueError("Feature and controller channels must be positive.")
        if deformable_groups != 8:
            raise ValueError("FaPN-Prefusion requires deformable_groups=8.")
        if high_channels % deformable_groups:
            raise ValueError(
                f"high_channels={high_channels} must be divisible by "
                f"deformable_groups={deformable_groups}."
            )

        self.high_channels = high_channels
        self.controller_channels = controller_channels
        self.deformable_groups = deformable_groups
        self.offset_channels = 2 * deformable_groups * self.kernel_size * self.kernel_size
        self.mask_channels = deformable_groups * self.kernel_size * self.kernel_size

        self.dcn = DeformConv2d(
            in_channels=high_channels,
            out_channels=high_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            dilation=1,
            groups=high_channels,
            bias=True,
        )
        self.conv_offset_mask = nn.Conv2d(
            controller_channels,
            self.offset_channels + self.mask_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Make the modulated depthwise convolution an exact initial identity."""

        with torch.no_grad():
            self.dcn.weight.zero_()
            self.dcn.weight[:, 0, 1, 1] = 2.0
            self.dcn.bias.zero_()
            self.conv_offset_mask.weight.zero_()
            self.conv_offset_mask.bias.zero_()

    def offset_and_mask(self, controller: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict 8-group offsets and modulation masks."""

        if controller.ndim != 4 or controller.shape[1] != self.controller_channels:
            raise ValueError(
                f"Expected controller [B,{self.controller_channels},H,W], "
                f"got {tuple(controller.shape)}."
            )
        logits = self.conv_offset_mask(controller)
        offset_y, offset_x, mask_logits = torch.chunk(logits, 3, dim=1)
        offset = torch.cat((offset_y, offset_x), dim=1)
        mask = torch.sigmoid(mask_logits)
        return offset, mask

    def forward(self, high: torch.Tensor, controller: torch.Tensor) -> torch.Tensor:
        """Align the complete high-level feature; no content bottleneck is used."""

        if high.ndim != 4 or high.shape[1] != self.high_channels:
            raise ValueError(
                f"Expected high feature [B,{self.high_channels},H,W], got {tuple(high.shape)}."
            )
        if high.shape[0] != controller.shape[0] or high.shape[-2:] != controller.shape[-2:]:
            raise ValueError("High feature and controller must share batch and spatial dimensions.")
        offset, mask = self.offset_and_mask(controller)
        if offset.shape[1] != self.offset_channels or mask.shape[1] != self.mask_channels:
            raise RuntimeError("Offset/mask channels do not satisfy Torchvision DCNv2 rules.")
        return self.dcn(high, offset, mask)


class FaPNAlignmentOnly(nn.Module):
    """Align ``upsampled_high`` to ``selected_low`` without performing fusion."""

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        controller_channels: int = 64,
        deformable_groups: int = 8,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if low_channels <= 0 or high_channels <= 0 or controller_channels <= 0:
            raise ValueError("All FaPN-Prefusion channel counts must be positive.")
        if high_channels % deformable_groups:
            raise ValueError(
                f"high_channels={high_channels} must be divisible by {deformable_groups}."
            )
        self.low_channels = low_channels
        self.high_channels = high_channels
        self.out_channels = high_channels
        self.controller_channels = controller_channels
        self.deformable_groups = deformable_groups

        self.low_projection = nn.Conv2d(low_channels, controller_channels, kernel_size=1, bias=False)
        self.high_projection = nn.Conv2d(high_channels, controller_channels, kernel_size=1, bias=False)
        self.offset_feature = nn.Conv2d(
            controller_channels * 2,
            controller_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dcn = FaPNDepthwiseModulatedDeformConv2d(
            high_channels,
            controller_channels=controller_channels,
            deformable_groups=deformable_groups,
        )
        self.gamma_a = nn.Parameter(torch.tensor(float(gamma_init)))
        _xavier_init(self.low_projection)
        _xavier_init(self.high_projection)
        _xavier_init(self.offset_feature)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return the residualized, aligned high feature in its original channels."""

        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("FaPNAlignmentOnly expects [selected_low, upsampled_high].")
        selected_low, upsampled_high = inputs
        if selected_low.ndim != 4 or selected_low.shape[1] != self.low_channels:
            raise ValueError(
                f"Expected selected_low [B,{self.low_channels},H,W], "
                f"got {tuple(selected_low.shape)}."
            )
        if upsampled_high.ndim != 4 or upsampled_high.shape[1] != self.high_channels:
            raise ValueError(
                f"Expected upsampled_high [B,{self.high_channels},H,W], "
                f"got {tuple(upsampled_high.shape)}."
            )
        if selected_low.shape[0] != upsampled_high.shape[0]:
            raise ValueError("Low and high features must share the batch dimension.")
        if selected_low.shape[-2:] != upsampled_high.shape[-2:]:
            raise ValueError(
                "FaPNAlignmentOnly does not upsample internally; inputs must already "
                "have identical spatial dimensions."
            )

        q_low = self.low_projection(selected_low)
        q_high = self.high_projection(upsampled_high)
        controller = self.offset_feature(torch.cat((q_low, q_high * 2.0), dim=1))
        aligned = self.dcn(upsampled_high, controller)
        return upsampled_high + self.gamma_a * (aligned - upsampled_high)
