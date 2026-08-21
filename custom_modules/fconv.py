"""Paper-derived FConv adaptation for YOLO11 feature pyramids.

The implementation follows the AMFC-DEIM description: Dynamic Tanh is
followed by three dilated depthwise branches (3x1, 1x3, and 3x3), branch-wise
ASR batch normalization, summation, and squeeze-excitation recalibration.
There is no public author implementation of FConv, so the equations and
Figure 5 in the paper are the implementation authority here.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping

import torch
import torch.nn as nn


class DynamicTanh2d(nn.Module):
    """Channel-wise Dynamic Tanh normalization from Transformers Without Normalization."""

    def __init__(self, channels: int, alpha_init: float = 0.5) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.tanh(self.alpha * x) + self.bias


class _SEGate(nn.Module):
    """Squeeze-excitation gate used by both ASR-BN and the output SE block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))


class _SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.gate = _SEGate(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ASRBatchNorm2d(nn.Module):
    """Attention-alike structural re-parameterization batch normalization.

    The attention input is a learned, data-independent channel vector as in
    the official ASR implementation. Consequently its output can be folded
    into the branch BN parameters for deployment.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.vector = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.attention = _SEGate(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x) * self.attention(self.vector)

    def equivalent_scale_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the affine scale and bias represented by ASR-BN in eval mode."""
        attention = self.attention(self.vector).reshape(-1)
        std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale = self.bn.weight / std * attention
        bias = (self.bn.bias - self.bn.running_mean * self.bn.weight / std) * attention
        return scale, bias


class FConv(nn.Module):
    """Focal convolution with a re-parameterizable effective ``K x K`` kernel."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 13,
        reduction: int = 16,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}.")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")

        self.channels = int(channels)
        self.kernel_size = kernel_size
        self.dilation = kernel_size // 2
        self.deploy = bool(deploy)
        self.dyt = DynamicTanh2d(self.channels)
        self.se = _SqueezeExcitation(self.channels, reduction)

        if self.deploy:
            self.reparam = nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size,
                stride=1,
                padding=self.dilation,
                groups=self.channels,
                bias=True,
            )
        else:
            d = self.dilation
            self.vertical = nn.Conv2d(
                self.channels,
                self.channels,
                (3, 1),
                padding=(d, 0),
                dilation=(d, 1),
                groups=self.channels,
                bias=False,
            )
            self.horizontal = nn.Conv2d(
                self.channels,
                self.channels,
                (1, 3),
                padding=(0, d),
                dilation=(1, d),
                groups=self.channels,
                bias=False,
            )
            self.spatial = nn.Conv2d(
                self.channels,
                self.channels,
                3,
                padding=d,
                dilation=d,
                groups=self.channels,
                bias=False,
            )
            self.vertical_norm = ASRBatchNorm2d(self.channels, reduction)
            self.horizontal_norm = ASRBatchNorm2d(self.channels, reduction)
            self.spatial_norm = ASRBatchNorm2d(self.channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dyt(x)
        if self.deploy:
            x = self.reparam(x)
        else:
            x = (
                self.vertical_norm(self.vertical(x))
                + self.horizontal_norm(self.horizontal(x))
                + self.spatial_norm(self.spatial(x))
            )
        return self.se(x)

    def _expand_dilated_kernel(self, kernel: torch.Tensor) -> torch.Tensor:
        """Expand a dilated 3x1/1x3/3x3 kernel into its dense effective KxK grid."""
        _, _, kh, kw = kernel.shape
        expanded = kernel.new_zeros((self.channels, 1, self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        for row in range(kh):
            target_row = center + (row - kh // 2) * self.dilation
            for col in range(kw):
                target_col = center + (col - kw // 2) * self.dilation
                expanded[:, :, target_row, target_col] = kernel[:, :, row, col]
        return expanded

    def _fuse_branch(self, conv: nn.Conv2d, norm: ASRBatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        scale, bias = norm.equivalent_scale_bias()
        kernel = conv.weight * scale.reshape(-1, 1, 1, 1)
        return self._expand_dilated_kernel(kernel), bias

    @torch.no_grad()
    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the single depthwise convolution equivalent to the three eval branches."""
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        branches = (
            self._fuse_branch(self.vertical, self.vertical_norm),
            self._fuse_branch(self.horizontal, self.horizontal_norm),
            self._fuse_branch(self.spatial, self.spatial_norm),
        )
        kernel = sum(item[0] for item in branches)
        bias = sum(item[1] for item in branches)
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self, delete_training_branches: bool = True) -> None:
        """Fold all depthwise branches and ASR-BN layers into one KxK depthwise convolution."""
        if self.deploy:
            return
        if self.training:
            raise RuntimeError("Call eval() before converting FConv to its deployment form.")
        kernel, bias = self.equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            stride=1,
            padding=self.dilation,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        reparam.weight.copy_(kernel)
        reparam.bias.copy_(bias)
        self.reparam = reparam
        self.deploy = True
        if delete_training_branches:
            for name in (
                "vertical",
                "horizontal",
                "spatial",
                "vertical_norm",
                "horizontal_norm",
                "spatial_norm",
            ):
                delattr(self, name)


def remap_yolo11n_state_dict_for_fconv(
    state_dict: Mapping[str, torch.Tensor],
    first_head_index: int = 11,
    inserted_layers: int = 3,
) -> OrderedDict[str, torch.Tensor]:
    """Shift baseline head indices after the three FConv pyramid layers are inserted.

    The FConv YAML preserves layers 0-10 and inserts three processors before the
    original head. This helper keeps pretrained transfer controlled by mapping
    every original ``model.11+`` tensor to ``model.14+`` while leaving the
    backbone keys unchanged.
    """
    remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    pattern = re.compile(r"^(.*model\.)(\d+)(.*)$")

    def shifted_key(key: str) -> str:
        match = pattern.match(key)
        if match and int(match.group(2)) >= first_head_index:
            index = int(match.group(2)) + inserted_layers
            return f"{match.group(1)}{index}{match.group(3)}"
        return key

    for key, value in state_dict.items():
        remapped[shifted_key(key)] = value
    if hasattr(state_dict, "_metadata"):
        remapped._metadata = OrderedDict(  # type: ignore[attr-defined]
            (shifted_key(key), value) for key, value in state_dict._metadata.items()
        )
    return remapped
