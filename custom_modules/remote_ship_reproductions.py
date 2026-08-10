"""Paper-level module reproductions for remote-sensing ship comparisons.

The complete detector implementations in the four comparison papers were not
released by their authors.  This file therefore follows the published network
figures and equations, while adapting reusable components from their official
module repositories when a compatible license is available:

* PKINet (Apache-2.0): https://github.com/PKINet/PKINet
* RepGhost (MIT): https://github.com/ChengpengChen/RepGhost
* DAT (Apache-2.0): https://github.com/LeapLabTHU/DAT

RFAConv, FASFF and the weighted fusion graph are independent implementations
from the equations in the corresponding detector papers.  They do not copy
code from repositories without an explicit license.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules import C2f, Conv, SPPF


def _valid_heads(channels: int, preferred: int = 8) -> int:
    """Return the largest practical attention-head count dividing channels."""

    for heads in range(min(preferred, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Resize a feature map to the reference spatial resolution."""

    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class ContextAnchorAttention(nn.Module):
    """Context-anchor attention used by the APFAN C3K2PKI block."""

    def __init__(self, channels: int, reduction: int = 4, strip_kernel: int = 11) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.horizontal = nn.Conv2d(
            hidden,
            hidden,
            (1, strip_kernel),
            padding=(0, strip_kernel // 2),
            groups=hidden,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            hidden,
            hidden,
            (strip_kernel, 1),
            padding=(strip_kernel // 2, 0),
            groups=hidden,
            bias=False,
        )
        self.expand = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.reduce(self.pool(x))
        attention = self.horizontal(attention)
        attention = self.vertical(attention)
        return torch.sigmoid(self.expand(attention))


class PolyKernelInception(nn.Module):
    """Dense, non-dilated 3/5/7/9/11 depthwise kernel mixer from PKINet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.base = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.branches = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel,
                1,
                kernel // 2,
                groups=channels,
                bias=False,
            )
            for kernel in (5, 7, 9, 11)
        )
        self.project = Conv(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        mixed = base
        for branch in self.branches:
            mixed = mixed + branch(base)
        return x + self.project(mixed)


class APFANPKIConv(nn.Module):
    """PKI convolution followed by context-anchor modulation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pki = PolyKernelInception(channels)
        self.caa = ContextAnchorAttention(channels)
        self.out = Conv(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.pki(x)
        return x + self.out(enhanced * self.caa(enhanced))


class APFANBottleneck(nn.Module):
    """Keep the first YOLO bottleneck convolution and replace only the second."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        if g != 1:
            raise ValueError("APFANBottleneck currently requires g=1.")
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 3, 1)
        self.adapter = Conv(hidden, c2, 1, 1) if hidden != c2 else nn.Identity()
        self.cv2 = APFANPKIConv(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.adapter(self.cv1(x)))
        return x + y if self.add else y


class APFANC3k(nn.Module):
    """C3k-compatible two-bottleneck branch using APFAN PKI convolution."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1) -> None:
        super().__init__()
        hidden = int(c2 * 0.5)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1, 1)
        self.m = nn.Sequential(
            APFANBottleneck(hidden, hidden, shortcut=shortcut, g=g, e=1.0),
            APFANBottleneck(hidden, hidden, shortcut=shortcut, g=g, e=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3k2APFAN(C2f):
    """C3K2PKI reproduction used at all four APFAN backbone stages."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        if attn:
            raise ValueError("C3k2APFAN does not add an internal PSA branch.")
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            APFANC3k(self.c, self.c, shortcut=shortcut, g=g)
            if c3k
            else APFANBottleneck(self.c, self.c, shortcut=shortcut, g=g)
            for _ in range(n)
        )


class CAFM(nn.Module):
    """Convolution-attention fusion module from the APFAN backbone figure."""

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        heads = _valid_heads(channels, heads)
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5

        self.local_reduce = Conv(channels, channels, 1, 1)
        self.local_mix = Conv(channels, channels, 3, 1)
        self.qkv = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, channels, 1, bias=False),
            )
            for _ in range(3)
        )
        self.global_project = Conv(channels, channels, 1, 1)
        self.out = Conv(channels, channels, 1, 1)

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
        b, c, h, w = x.shape
        if c % groups:
            return x
        return x.view(b, groups, c // groups, h, w).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        local = self.local_mix(self._channel_shuffle(self.local_reduce(x)))
        q, k, v = (projection(x) for projection in self.qkv)
        q = q.view(b, self.heads, self.head_dim, h * w).transpose(-2, -1)
        k = k.view(b, self.heads, self.head_dim, h * w)
        v = v.view(b, self.heads, self.head_dim, h * w).transpose(-2, -1)
        attention = (q @ k * self.scale).softmax(dim=-1)
        global_feature = (attention @ v).transpose(-2, -1)
        global_feature = global_feature.reshape(b, self.channels, h, w)
        global_feature = self.global_project(global_feature) + x
        return self.out(local + global_feature)


class AMSFA(nn.Module):
    """Adaptive multi-scale feature aggregation at a selected pyramid level."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        target_index: int = 1,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("AMSFA requires exactly P3, P4 and P5 inputs.")
        if target_index not in range(3):
            raise ValueError(f"target_index must be 0, 1 or 2, got {target_index}.")
        self.target_index = target_index
        self.projections = nn.ModuleList(Conv(c, out_channels, 1, 1) for c in in_channels)
        mixed_channels = out_channels * 3
        self.kernels = nn.ModuleList(
            nn.Conv2d(
                mixed_channels,
                mixed_channels,
                kernel,
                1,
                kernel // 2,
                groups=mixed_channels,
                bias=False,
            )
            for kernel in (5, 7, 9, 11)
        )
        self.enhance = Conv(mixed_channels, out_channels, 1, 1)
        hidden = max(out_channels // 4, 8)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 3:
            raise ValueError(f"AMSFA expected 3 tensors, got {len(inputs)}.")
        reference = inputs[self.target_index]
        aligned = [
            _resize_like(projection(feature), reference)
            for projection, feature in zip(self.projections, inputs)
        ]
        original = aligned[self.target_index]
        fused = torch.cat(aligned, dim=1)
        mixed = fused
        for kernel in self.kernels:
            mixed = mixed + kernel(fused)
        enhanced = self.enhance(mixed)
        enhanced = enhanced * self.channel_attention(enhanced)

        enhanced_energy = enhanced.square().mean((1, 2, 3), keepdim=True)
        original_energy = original.square().mean((1, 2, 3), keepdim=True)
        normalizer = enhanced_energy + original_energy + 1e-6
        alpha = enhanced_energy / normalizer
        beta = original_energy / normalizer
        return alpha * enhanced + beta * original


class SqueezeExcite(nn.Module):
    """Squeeze-excitation layer used in the SHIP-YOLO RepGhost bottleneck."""

    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(int(channels * ratio), 4)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.reduce(scale), inplace=True)
        scale = F.hardsigmoid(self.expand(scale))
        return x * scale


class RepGhostModule(nn.Module):
    """Training-time RepGhost module adapted from the official MIT code."""

    def __init__(self, c1: int, c2: int, relu: bool = True) -> None:
        super().__init__()
        self.primary = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.reparameterization_bn = nn.BatchNorm2d(c2)
        self.activation = nn.ReLU(inplace=False) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        return self.activation(self.cheap(primary) + self.reparameterization_bn(primary))


class SHIPRepGhostBottleneck(nn.Module):
    """RepGhost bottleneck described in the SHIP-YOLO paper."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True) -> None:
        super().__init__()
        hidden = max(c2 // 2, 8)
        self.reduce = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.repghost = RepGhostModule(hidden, hidden, relu=True)
        self.se = SqueezeExcite(hidden)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.se(self.repghost(self.reduce(x))))
        return x + y if self.add else y


class C2fRepGhost(C2f):
    """YOLOv8 C2f with each Bottleneck replaced by SHIP RepGhost."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        if g != 1:
            raise ValueError("C2fRepGhost requires g=1.")
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            SHIPRepGhostBottleneck(self.c, self.c, shortcut=shortcut)
            for _ in range(n)
        )


class ShuffleAttention(nn.Module):
    """Shuffle Attention (SA-Net, ICASSP 2021) with robust group selection."""

    def __init__(self, channels: int, groups: int = 64) -> None:
        super().__init__()
        groups = min(groups, max(channels // 2, 1))
        while groups > 1 and channels % (2 * groups):
            groups -= 1
        self.groups = groups
        branch_channels = channels // (2 * groups)
        self.channel_weight = nn.Parameter(torch.zeros(1, branch_channels, 1, 1))
        self.channel_bias = nn.Parameter(torch.ones(1, branch_channels, 1, 1))
        self.spatial_weight = nn.Parameter(torch.zeros(1, branch_channels, 1, 1))
        self.spatial_bias = nn.Parameter(torch.ones(1, branch_channels, 1, 1))
        self.norm = nn.GroupNorm(branch_channels, branch_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        grouped = x.reshape(b * self.groups, -1, h, w)
        channel, spatial = grouped.chunk(2, dim=1)
        channel_scale = F.adaptive_avg_pool2d(channel, 1)
        channel = channel * torch.sigmoid(
            self.channel_weight * channel_scale + self.channel_bias
        )
        spatial = spatial * torch.sigmoid(
            self.spatial_weight * self.norm(spatial) + self.spatial_bias
        )
        merged = torch.cat((channel, spatial), dim=1).reshape(b, c, h, w)
        return (
            merged.reshape(b, 2, c // 2, h, w)
            .transpose(1, 2)
            .reshape(b, c, h, w)
        )


class RFAConv(nn.Module):
    """Receptive-field attention convolution independently implemented from equations."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("RFAConv requires an odd kernel size.")
        self.kernel_size = kernel_size
        area = kernel_size * kernel_size
        self.weight_generator = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2),
            nn.Conv2d(c1, c1 * area, 1, groups=c1, bias=False),
        )
        self.feature_generator = nn.Sequential(
            nn.Conv2d(
                c1,
                c1 * area,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=c1,
                bias=False,
            ),
            nn.BatchNorm2d(c1 * area),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size, stride=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.kernel_size
        area = k * k
        weights = self.weight_generator(x).view(b, c, area, h, w).softmax(dim=2)
        features = self.feature_generator(x).view(b, c, area, h, w)
        weighted = (weights * features).view(b, c, k, k, h, w)
        expanded = weighted.permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * k, w * k)
        return self.project(expanded)


class RFABottleneck(nn.Module):
    """Preserve C2f Bottleneck Conv1 and replace Conv2 with RFAConv."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 1.0,
    ) -> None:
        super().__init__()
        if g != 1:
            raise ValueError("RFABottleneck requires g=1.")
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 3, 1)
        self.cv2 = RFAConv(hidden, c2, kernel_size=3)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2fRFA(C2f):
    """R-C2f used only in the PMF-YOLOv8 backbone."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            RFABottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0)
            for _ in range(n)
        )


class FASFF(nn.Module):
    """Four-level feature-adaptive spatial fusion used before PMF heads."""

    def __init__(self, in_channels: Sequence[int], target_index: int) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("FASFF requires four PAN-4 inputs.")
        if target_index not in range(4):
            raise ValueError(f"target_index must be in [0, 3], got {target_index}.")
        self.target_index = target_index
        out_channels = int(in_channels[target_index])
        # The paper's feature-resizing stage uses learned spatial transforms,
        # not parameter-free interpolation alone. A 3x3 projection is applied
        # at every source level before resolution alignment.
        self.projections = nn.ModuleList(Conv(c, out_channels, 3, 1) for c in in_channels)
        self.weight_logits = nn.ModuleList(
            nn.Conv2d(out_channels, 1, 1) for _ in in_channels
        )
        self.out = nn.Sequential(
            Conv(out_channels, out_channels, 3, 1),
            Conv(out_channels, out_channels, 3, 1),
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 4:
            raise ValueError(f"FASFF expected 4 tensors, got {len(inputs)}.")
        reference = inputs[self.target_index]
        aligned = [
            _resize_like(projection(feature), reference)
            for projection, feature in zip(self.projections, inputs)
        ]
        logits = torch.cat(
            [head(feature) for head, feature in zip(self.weight_logits, aligned)],
            dim=1,
        )
        weights = logits.softmax(dim=1)
        fused = sum(
            feature * weights[:, index : index + 1]
            for index, feature in enumerate(aligned)
        )
        return self.out(fused)


class LayerNorm2d(nn.Module):
    """Channel-first layer normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DATBlock(nn.Module):
    """Deformable-attention block adapted from the official DAT formulation."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        sample_stride: int = 2,
        offset_range: float = 2.0,
    ) -> None:
        super().__init__()
        heads = _valid_heads(channels, heads)
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.sample_stride = sample_stride
        self.offset_range = offset_range
        self.norm1 = LayerNorm2d(channels)
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.offset_stem = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                5,
                stride=sample_stride,
                padding=2,
                groups=channels,
                bias=False,
            ),
            nn.GELU(),
        )
        self.offset_heads = nn.ModuleList(
            nn.Conv2d(channels, 2, 1, bias=True) for _ in range(3)
        )
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.relative_bias = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, heads),
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = LayerNorm2d(channels)
        hidden = channels * 2
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    @staticmethod
    def _grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((x, y), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm1(x)
        b, _, h, w = normalized.shape
        offset_feature = self.offset_stem(normalized)
        hs, ws = offset_feature.shape[-2:]
        offsets = torch.stack(
            [head(offset_feature) for head in self.offset_heads], dim=0
        ).mean(dim=0)
        offsets = torch.tanh(offsets).permute(0, 2, 3, 1)
        offset_scale = offsets.new_tensor(
            [
                self.offset_range / max(ws, 1),
                self.offset_range / max(hs, 1),
            ]
        )
        offsets = offsets * offset_scale
        sample_grid = self._grid(hs, ws, x.device, x.dtype).unsqueeze(0) + offsets
        sampled = F.grid_sample(
            normalized,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        q = self.query(normalized).view(b, self.heads, self.head_dim, h * w).transpose(-2, -1)
        k = self.key(sampled).view(b, self.heads, self.head_dim, hs * ws)
        v = self.value(sampled).view(b, self.heads, self.head_dim, hs * ws).transpose(-2, -1)

        query_grid = self._grid(h, w, x.device, x.dtype).reshape(h * w, 1, 2)
        key_grid = sample_grid.mean(dim=0).reshape(1, hs * ws, 2)
        relative = query_grid - key_grid
        bias = self.relative_bias(relative).permute(2, 0, 1).unsqueeze(0)
        attention = (q @ k * self.scale + bias).softmax(dim=-1)
        attended = (attention @ v).transpose(-2, -1)
        attended = attended.reshape(b, self.channels, h, w)
        x = residual + self.project(attended)
        return x + self.ffn(self.norm2(x))


class SimSPPF(SPPF):
    """Named SPPF variant shown in E-WFF Net (same sequential-pooling topology)."""


class WeightedFeatureFusion(nn.Module):
    """Fast normalized residual fusion from E-WFF equations (11)-(13)."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        target_index: int = 0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if not in_channels:
            raise ValueError("WeightedFeatureFusion requires at least one input.")
        if target_index not in range(len(in_channels)):
            raise ValueError("target_index is outside the input list.")
        self.target_index = target_index
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(len(in_channels), dtype=torch.float32))
        self.projections = nn.ModuleList(Conv(c, out_channels, 1, 1) for c in in_channels)
        self.out = Conv(out_channels, out_channels, 3, 1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != len(self.projections):
            raise ValueError(
                f"WeightedFeatureFusion expected {len(self.projections)} inputs, got {len(inputs)}."
            )
        reference = inputs[self.target_index]
        aligned = [
            _resize_like(projection(feature), reference)
            for projection, feature in zip(self.projections, inputs)
        ]
        positive = F.relu(self.weights)
        normalized = positive / (positive.sum() + self.epsilon)
        fused = sum(weight * feature for weight, feature in zip(normalized, aligned))
        return self.out(fused)
