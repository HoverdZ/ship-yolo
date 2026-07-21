"""Official FaPN top-down building blocks ported to PyTorch/Torchvision.

Core design adapted from:
FaPN: Feature-Aligned Pyramid Network for Dense Image Prediction, ICCV 2021
Official implementation: EMI-Group/FaPN, detectron2/modeling/backbone/fan.py
Original DCNv2 reference: EMI-Group/FaPN, DCNv2/dcn_v2.py
License: Apache-2.0

This file keeps the published FSM/FAM computation while replacing only the
legacy custom CUDA DCNv2 binding with Torchvision's equivalent modulated
deformable-convolution operator. It does not add normalization, attention,
gating, or lightweight substitutions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d


def _xavier_init(module: nn.Conv2d) -> None:
    """Apply the requested Xavier initialization and zero any bias."""

    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class FaPNFeatureSelection(nn.Module):
    """Official FeatureSelectionModule without Detectron2 wrappers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("FaPN FSM channel counts must be positive.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        # FeatureAlign_V2 calls the official FSM with norm="", so neither
        # convolution has normalization and both keep bias=False.
        self.conv_attention = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        _xavier_init(self.conv_attention)
        _xavier_init(self.projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute projection(x + x * sigmoid(conv1x1(GAP(x))))."""

        if x.ndim != 4:
            raise ValueError(f"FaPN FSM expects BCHW input, got {tuple(x.shape)}.")
        attention = self.sigmoid(self.conv_attention(F.adaptive_avg_pool2d(x, output_size=1)))
        selected = x * attention
        enhanced = x + selected
        return self.projection(enhanced)


class FaPNModulatedDeformConv2d(nn.Module):
    """Torchvision equivalent of the official FaPN modulated DCNv2 wrapper."""

    kernel_size = 3

    def __init__(self, channels: int, deformable_groups: int = 8) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"FaPN DCNv2 channels must be positive, got {channels}.")
        if deformable_groups != 8:
            raise ValueError("The original FaPN experiment requires deformable_groups=8.")
        if channels % deformable_groups != 0:
            raise ValueError(
                f"FaPN channels ({channels}) must be divisible by deformable_groups "
                f"({deformable_groups})."
            )

        self.channels = channels
        self.deformable_groups = deformable_groups
        self.offset_channels = 2 * deformable_groups * self.kernel_size * self.kernel_size
        self.mask_channels = deformable_groups * self.kernel_size * self.kernel_size
        self.dcn = DeformConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
        )
        # Match the legacy DCNv2 parameter reset exactly: uniform
        # +/-1/sqrt(C*3*3) for weights and zero bias.
        bound = 1.0 / math.sqrt(channels * self.kernel_size * self.kernel_size)
        nn.init.uniform_(self.dcn.weight, -bound, bound)
        nn.init.zeros_(self.dcn.bias)
        self.conv_offset_mask = nn.Conv2d(
            channels,
            self.offset_channels + self.mask_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            bias=True,
        )
        # Official DCNv2 initializes both tensors to zero: initial offsets are
        # zero and sigmoid(mask_logits) is exactly 0.5.
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def offset_and_mask(self, offset_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate offsets and modulation masks using official channel rules."""

        if offset_feature.ndim != 4 or offset_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected offset feature [B,{self.channels},H,W], got {tuple(offset_feature.shape)}."
            )
        offset_mask = self.conv_offset_mask(offset_feature)
        offset_y, offset_x, mask_logits = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_y, offset_x), dim=1)
        mask = torch.sigmoid(mask_logits)
        return offset, mask

    def forward(self, input_feature: torch.Tensor, offset_feature: torch.Tensor) -> torch.Tensor:
        """Apply modulated deformable convolution to the top-down feature."""

        if input_feature.ndim != 4 or input_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected DCN input [B,{self.channels},H,W], got {tuple(input_feature.shape)}."
            )
        if input_feature.shape[0] != offset_feature.shape[0] or input_feature.shape[-2:] != offset_feature.shape[-2:]:
            raise ValueError("DCN input and offset feature must share batch and spatial dimensions.")
        offset, mask = self.offset_and_mask(offset_feature)
        if offset.shape[1] != self.offset_channels or mask.shape[1] != self.mask_channels:
            raise RuntimeError("Generated FaPN offset/mask channels do not follow DCNv2 rules.")
        return self.dcn(input_feature, offset, mask)


class FaPNAlign(nn.Module):
    """Official FeatureAlign_V2 adapted to a YOLO list-input module."""

    def __init__(
        self,
        lateral_channels: int,
        topdown_channels: int,
        out_channels: int | None = None,
        deformable_groups: int = 8,
    ) -> None:
        super().__init__()
        # The two-argument form mirrors official FeatureAlign_V2(in_nc,
        # out_nc); the explicit three-argument form is used by YAML parsing
        # so the top-down input channel count is still audited.
        out_channels = topdown_channels if out_channels is None else out_channels
        if topdown_channels != out_channels:
            raise ValueError(
                "Official FaPN recursion requires the top-down input and output to use "
                f"the same channels; got {topdown_channels} and {out_channels}."
            )
        if out_channels % deformable_groups != 0:
            raise ValueError(
                f"FaPN out_channels ({out_channels}) must be divisible by {deformable_groups}."
            )
        self.lateral_channels = lateral_channels
        self.topdown_channels = topdown_channels
        self.out_channels = out_channels
        self.fsm = FaPNFeatureSelection(lateral_channels, out_channels)
        self.offset_feature = nn.Conv2d(
            out_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        _xavier_init(self.offset_feature)
        self.dcn = FaPNModulatedDeformConv2d(out_channels, deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        inputs: Sequence[torch.Tensor] | torch.Tensor,
        feat_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Align top-down ``feat_s`` to shallow ``feat_l`` and add the FSM path."""

        if feat_s is None:
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
                raise ValueError("FaPNAlign expects [feat_l, feat_s].")
            feat_l, feat_s = inputs
        else:
            if not isinstance(inputs, torch.Tensor):
                raise ValueError("Direct FaPNAlign calls require tensor feat_l and feat_s.")
            feat_l = inputs

        if feat_l.ndim != 4 or feat_l.shape[1] != self.lateral_channels:
            raise ValueError(
                f"Expected lateral feature [B,{self.lateral_channels},H,W], got {tuple(feat_l.shape)}."
            )
        if feat_s.ndim != 4 or feat_s.shape[1] != self.topdown_channels:
            raise ValueError(
                f"Expected top-down feature [B,{self.topdown_channels},H,W], got {tuple(feat_s.shape)}."
            )

        target_size = feat_l.shape[-2:]
        feat_up = F.interpolate(feat_s, size=target_size, mode="bilinear", align_corners=False)
        feat_arm = self.fsm(feat_l)
        offset_source = self.offset_feature(torch.cat((feat_arm, feat_up * 2.0), dim=1))
        aligned = self.relu(self.dcn(input_feature=feat_up, offset_feature=offset_source))
        return aligned + feat_arm


class FaPNLateral(nn.Module):
    """C5 1x1 lateral projection that produces the initial M5 feature."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        if c2 % 8 != 0:
            raise ValueError(f"FaPN lateral output channels ({c2}) must be divisible by 8.")
        self.in_channels = c1
        self.out_channels = c2
        # The official FAN code leaves this Detectron2 Conv2d on its framework
        # default initialization; nn.Conv2d preserves that behavior.
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FaPNOutputConv(nn.Module):
    """Official per-level 3x3 output convolution (M4/M3 -> T4/T3)."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError(f"Official FaPN output conv requires c1 == c2, got {c1} and {c2}.")
        if c2 % 8 != 0:
            raise ValueError(f"FaPN output channels ({c2}) must be divisible by 8.")
        self.in_channels = c1
        self.out_channels = c2
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=True)
        _xavier_init(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
