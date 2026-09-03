"""Controlled gate-generation studies for VGUP.

The BPW/KBL filters and their parameter prediction path are intentionally
identical to VGUP.  Attention modules only refine the shared 128-channel,
low-resolution encoder feature used to generate residual-acceptance gates.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from custom_modules.erup import (
    BPWFilter,
    BPW_PARAMETER_COUNT,
    KBLFilter,
    KBL_PARAMETER_COUNT,
    _require_image,
    _tensor_stats,
)
from custom_modules.remote_ship_reproductions import ShuffleAttention
from custom_modules.vgup import DepthwiseSeparableConv


ENCODER_CHANNELS = 128
GATE_PARAMETER_LIMIT = 7_714
GLOBAL_GATE_NAMES = frozenset({"none", "eca", "gct", "se", "ge"})
SPATIAL_GATE_NAMES = frozenset(
    {"none", "simam", "sge", "cbam", "coordatt", "triplet"}
)
JOINT_GATE_NAMES = frozenset({"none", "shuffle"})


def _normalize_gate_name(
    value: str,
    *,
    argument: str,
    choices: frozenset[str],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{argument} must be a string, got {type(value).__name__}.")
    normalized = value.strip().lower()
    if normalized not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"Unsupported {argument}={value!r}; choose one of: {allowed}.")
    return normalized


def _compatible_groups(channels: int, requested: int) -> int:
    """Select the largest requested-or-smaller divisor of ``channels``."""

    if not isinstance(requested, int) or requested < 1:
        raise ValueError(f"groups must be a positive integer, got {requested!r}.")
    groups = min(channels, requested)
    while channels % groups:
        groups -= 1
    return groups


class ECAGate(nn.Module):
    """ECA-Net channel refinement using the official GAP/Conv1d design."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be a positive odd integer.")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        descriptor = self.pool(features)
        descriptor = descriptor.squeeze(-1).transpose(-1, -2)
        scale = self.conv(descriptor).transpose(-1, -2).unsqueeze(-1).sigmoid()
        return features * scale


class GCTGate(nn.Module):
    """Official L2 Gated Channel Transformation formulation."""

    def __init__(self, channels: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("GCT epsilon must be positive.")
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = float(epsilon)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedding = (
            features.square().sum(dim=(2, 3), keepdim=True) + self.epsilon
        ).sqrt() * self.alpha
        norm = self.gamma / (
            embedding.square().mean(dim=1, keepdim=True) + self.epsilon
        ).sqrt()
        gate = 1.0 + torch.tanh(embedding * norm + self.beta)
        return features * gate


class SEGate(nn.Module):
    """Standard squeeze-and-excitation translated from SENet (r=16)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("SE reduction must be positive.")
        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Linear(channels, hidden_channels)
        self.activate = nn.ReLU()
        self.expand = nn.Linear(hidden_channels, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        descriptor = self.pool(features).flatten(1)
        scale = self.expand(self.activate(self.reduce(descriptor))).sigmoid()
        return features * scale[:, :, None, None]


class GatherExciteGate(nn.Module):
    """Parameter-free global Gather-Excite (GEPF) refinement."""

    def __init__(self) -> None:
        super().__init__()
        self.gather = nn.AdaptiveAvgPool2d(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.gather(features).sigmoid()


class SimAMGate(nn.Module):
    """Parameter-free SimAM using the official closed-form energy function."""

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        if e_lambda <= 0:
            raise ValueError("SimAM e_lambda must be positive.")
        self.e_lambda = float(e_lambda)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_count = features.shape[-2] * features.shape[-1]
        normalizer = max(spatial_count - 1, 1)
        deviation = (
            features - features.mean(dim=(2, 3), keepdim=True)
        ).square()
        variance = deviation.sum(dim=(2, 3), keepdim=True) / normalizer
        energy = deviation / (4.0 * (variance + self.e_lambda)) + 0.5
        return features * energy.sigmoid()


class SpatialGroupEnhanceGate(nn.Module):
    """Spatial Group-wise Enhance with a safe, channel-divisible group count."""

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("SGE epsilon must be positive.")
        self.groups = _compatible_groups(channels, groups)
        self.epsilon = float(epsilon)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, self.groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, self.groups, 1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        grouped = features.reshape(
            batch * self.groups,
            channels // self.groups,
            height,
            width,
        )
        descriptor = (grouped * self.pool(grouped)).sum(dim=1, keepdim=True)
        flattened = descriptor.flatten(1)
        centered = flattened - flattened.mean(dim=1, keepdim=True)
        variance = centered.square().mean(dim=1, keepdim=True)
        normalized = centered / (variance + self.epsilon).sqrt()
        normalized = normalized.reshape(batch, self.groups, height, width)
        scale = (normalized * self.weight + self.bias).reshape(
            batch * self.groups,
            1,
            height,
            width,
        )
        return (grouped * scale.sigmoid()).reshape(
            batch,
            channels,
            height,
            width,
        )


class CBAMSpatialGate(nn.Module):
    """The official CBAM SpatialGate only (2-to-1 7x7 convolution)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        maximum = features.amax(dim=1, keepdim=True)
        average = features.mean(dim=1, keepdim=True)
        scale = self.norm(self.conv(torch.cat((maximum, average), dim=1))).sigmoid()
        return features * scale


class HardSigmoid(nn.Module):
    """Coordinate Attention's h-sigmoid without in-place operations."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu6(value + 3.0, inplace=False) / 6.0


class HardSwish(nn.Module):
    """Coordinate Attention's h-swish without in-place operations."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = HardSigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.gate(value)


class CoordinateAttentionGate(nn.Module):
    """Coordinate Attention with the official reduction and min-8 bottleneck."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("Coordinate Attention reduction must be positive.")
        hidden_channels = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden_channels, 1)
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.activate = HardSwish()
        self.height_expand = nn.Conv2d(hidden_channels, channels, 1)
        self.width_expand = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        height, width = features.shape[-2:]
        height_context = features.mean(dim=3, keepdim=True)
        width_context = features.mean(dim=2, keepdim=True).transpose(2, 3)
        context = torch.cat((height_context, width_context), dim=2)
        context = self.activate(self.norm(self.reduce(context)))
        height_context, width_context = torch.split(
            context,
            (height, width),
            dim=2,
        )
        width_context = width_context.transpose(2, 3)
        height_scale = self.height_expand(height_context).sigmoid()
        width_scale = self.width_expand(width_context).sigmoid()
        return features * height_scale * width_scale


class _ChannelPool(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        maximum = features.amax(dim=1, keepdim=True)
        average = features.mean(dim=1, keepdim=True)
        return torch.cat((maximum, average), dim=1)


class _TripletAttentionBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compress = _ChannelPool()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scale = self.norm(self.conv(self.compress(features))).sigmoid()
        return features * scale


class TripletAttentionGate(nn.Module):
    """Official three-branch cross-dimension Triplet Attention."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_width = _TripletAttentionBranch()
        self.height_channel = _TripletAttentionBranch()
        self.height_width = _TripletAttentionBranch()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_width = features.permute(0, 2, 1, 3).contiguous()
        channel_width = self.channel_width(channel_width)
        channel_width = channel_width.permute(0, 2, 1, 3).contiguous()

        height_channel = features.permute(0, 3, 2, 1).contiguous()
        height_channel = self.height_channel(height_channel)
        height_channel = height_channel.permute(0, 3, 2, 1).contiguous()

        height_width = self.height_width(features)
        return (channel_width + height_channel + height_width) / 3.0


def _build_global_gate(name: str, channels: int) -> nn.Module:
    builders = {
        "none": nn.Identity,
        "eca": ECAGate,
        "gct": lambda: GCTGate(channels),
        "se": lambda: SEGate(channels, reduction=16),
        "ge": GatherExciteGate,
    }
    return builders[name]()


def _build_spatial_gate(name: str, channels: int) -> nn.Module:
    builders = {
        "none": nn.Identity,
        "simam": SimAMGate,
        "sge": lambda: SpatialGroupEnhanceGate(channels, groups=8),
        "cbam": CBAMSpatialGate,
        "coordatt": lambda: CoordinateAttentionGate(channels, reduction=32),
        "triplet": TripletAttentionGate,
    }
    return builders[name]()


class VGUPGateStudyEncoder(nn.Module):
    """Shared lightweight encoder plus selectable low-resolution gate paths."""

    parameter_output_count = BPW_PARAMETER_COUNT + KBL_PARAMETER_COUNT

    def __init__(
        self,
        in_channels: int = 3,
        prediction_size: int = 128,
        global_gate: str = "none",
        spatial_gate: str = "none",
        joint_gate: str = "none",
        global_gate_bias: float = -1.5,
        spatial_gate_bias: float = -1.0,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("VGUPGateStudyEncoder requires RGB input.")
        if not isinstance(prediction_size, int) or prediction_size < 32:
            raise ValueError("prediction_size must be an integer of at least 32.")

        self.prediction_size = prediction_size
        self.global_gate_name = _normalize_gate_name(
            global_gate,
            argument="global_gate",
            choices=GLOBAL_GATE_NAMES,
        )
        self.spatial_gate_name = _normalize_gate_name(
            spatial_gate,
            argument="spatial_gate",
            choices=SPATIAL_GATE_NAMES,
        )
        self.joint_gate_name = _normalize_gate_name(
            joint_gate,
            argument="joint_gate",
            choices=JOINT_GATE_NAMES,
        )
        if self.joint_gate_name != "none" and (
            self.global_gate_name != "none" or self.spatial_gate_name != "none"
        ):
            raise ValueError(
                "joint_gate='shuffle' cannot be combined with global_gate or "
                "spatial_gate."
            )

        self.has_global_gate = (
            self.global_gate_name != "none" or self.joint_gate_name != "none"
        )
        self.has_spatial_gate = (
            self.spatial_gate_name != "none" or self.joint_gate_name != "none"
        )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, ENCODER_CHANNELS, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.filter_head = nn.Linear(
            ENCODER_CHANNELS,
            self.parameter_output_count,
        )

        self.global_attention = _build_global_gate(
            self.global_gate_name,
            ENCODER_CHANNELS,
        )
        self.spatial_attention = _build_spatial_gate(
            self.spatial_gate_name,
            ENCODER_CHANNELS,
        )
        self.joint_attention = (
            ShuffleAttention(ENCODER_CHANNELS)
            if self.joint_gate_name == "shuffle"
            else nn.Identity()
        )
        self.global_gate_head = (
            nn.Linear(ENCODER_CHANNELS, 1) if self.has_global_gate else None
        )
        self.spatial_gate_head = (
            nn.Conv2d(ENCODER_CHANNELS, 1, 1) if self.has_spatial_gate else None
        )

        nn.init.normal_(self.filter_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.filter_head.bias)
        if self.global_gate_head is not None:
            nn.init.normal_(self.global_gate_head.weight, mean=0.0, std=1e-4)
            nn.init.constant_(self.global_gate_head.bias, float(global_gate_bias))
        if self.spatial_gate_head is not None:
            nn.init.normal_(self.spatial_gate_head.weight, mean=0.0, std=1e-4)
            nn.init.constant_(self.spatial_gate_head.bias, float(spatial_gate_bias))

        gate_modules = (
            self.global_attention,
            self.spatial_attention,
            self.joint_attention,
            self.global_gate_head,
            self.spatial_gate_head,
        )
        self.gate_trainable_parameters = sum(
            parameter.numel()
            for module in gate_modules
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        if self.gate_trainable_parameters >= GATE_PARAMETER_LIMIT:
            raise ValueError(
                "Gate-specific trainable parameter count must be below "
                f"{GATE_PARAMETER_LIMIT:,}, got "
                f"{self.gate_trainable_parameters:,}."
            )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        _require_image(image)
        resized = F.interpolate(
            image,
            size=(self.prediction_size, self.prediction_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.blocks(self.stem(resized))
        pooled = self.pool(features).flatten(1)
        filter_params = self.filter_head(pooled).tanh()
        bpw_params, kbl_params = torch.split(
            filter_params,
            (BPW_PARAMETER_COUNT, KBL_PARAMETER_COUNT),
            dim=1,
        )

        if self.joint_gate_name == "shuffle":
            global_features = spatial_features = self.joint_attention(features)
        else:
            global_features = self.global_attention(features)
            spatial_features = self.spatial_attention(features)

        if self.global_gate_head is None:
            global_gate = image.new_ones((image.shape[0], 1, 1, 1))
        else:
            global_descriptor = self.pool(global_features).flatten(1)
            global_gate = self.global_gate_head(global_descriptor).sigmoid()
            global_gate = global_gate.reshape(image.shape[0], 1, 1, 1)

        if self.spatial_gate_head is None:
            spatial_gate_lowres = image.new_ones(
                (image.shape[0], 1, features.shape[-2], features.shape[-1])
            )
            spatial_gate = image.new_ones(
                (image.shape[0], 1, image.shape[-2], image.shape[-1])
            )
        else:
            spatial_gate_lowres = self.spatial_gate_head(spatial_features).sigmoid()
            spatial_gate = F.interpolate(
                spatial_gate_lowres,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return {
            "bpw_params": bpw_params,
            "kbl_params": kbl_params,
            "global_gate": global_gate,
            "spatial_gate": spatial_gate,
            "spatial_gate_lowres": spatial_gate_lowres,
        }


class VGUPGateStudyPreprocessor(nn.Module):
    """VGUP with independently selectable global, spatial, or joint gates."""

    out_channels = 3

    def __init__(
        self,
        in_channels: int = 3,
        bpw_segments: int = 8,
        prediction_size: int = 128,
        global_gate: str = "none",
        spatial_gate: str = "none",
        joint_gate: str = "none",
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(
                "VGUPGateStudyPreprocessor must be the first RGB model layer."
            )
        self.in_channels = in_channels
        self.encoder = VGUPGateStudyEncoder(
            in_channels=in_channels,
            prediction_size=prediction_size,
            global_gate=global_gate,
            spatial_gate=spatial_gate,
            joint_gate=joint_gate,
        )
        self.global_gate = self.encoder.global_gate_name
        self.spatial_gate = self.encoder.spatial_gate_name
        self.joint_gate = self.encoder.joint_gate_name
        self.bpw = BPWFilter(segments=bpw_segments)
        self.kbl = KBLFilter()

    def forward(
        self,
        image: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        predictions = self.encoder(image)
        enhanced_bpw = self.bpw(image, predictions["bpw_params"])
        if self.encoder.has_global_gate:
            image_bpw = image + predictions["global_gate"] * (
                enhanced_bpw - image
            )
        else:
            image_bpw = enhanced_bpw

        enhanced_kbl = self.kbl(image_bpw, predictions["kbl_params"])
        if self.encoder.has_spatial_gate:
            output = image_bpw + predictions["spatial_gate"] * (
                enhanced_kbl - image_bpw
            )
        else:
            output = enhanced_kbl

        if not return_debug:
            return output
        debug = {
            **predictions,
            "bpw_param_stats": _tensor_stats(predictions["bpw_params"]),
            "kbl_param_stats": _tensor_stats(predictions["kbl_params"]),
            "global_gate_stats": _tensor_stats(predictions["global_gate"]),
            "spatial_gate_stats": _tensor_stats(predictions["spatial_gate"]),
            "bpw_image": enhanced_bpw,
            "gated_bpw_image": image_bpw,
            "kbl_image": enhanced_kbl,
            "output_image": output,
        }
        return output, debug


__all__ = [
    "CBAMSpatialGate",
    "CoordinateAttentionGate",
    "ECAGate",
    "GATE_PARAMETER_LIMIT",
    "GCTGate",
    "GatherExciteGate",
    "SEGate",
    "SimAMGate",
    "SpatialGroupEnhanceGate",
    "TripletAttentionGate",
    "VGUPGateStudyEncoder",
    "VGUPGateStudyPreprocessor",
]
