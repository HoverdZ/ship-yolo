"""ASCGD-Neck building blocks implemented with standard PyTorch operators.

The modules in this file are deliberately single-output modules so they can be
referenced safely by the Ultralytics YAML parser.  The validated InceptionDW
backbone is not duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


GATHER_DIM = 128
HEADS = 4
WINDOW_SIZE = 8
SGFN_EXPANSION = 2.0
DUAL_AGGREGATION_BLOCKS = 1


def _check_bchw(x: Tensor, name: str) -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} must be BCHW, got shape {tuple(x.shape)}.")


def _check_feature_list(
    features: Sequence[Tensor],
    channels: Sequence[int],
    module_name: str,
) -> None:
    if not isinstance(features, (list, tuple)):
        raise TypeError(f"{module_name} expects a list/tuple of feature tensors.")
    if len(features) != len(channels):
        raise ValueError(
            f"{module_name} expects {len(channels)} inputs, got {len(features)}."
        )
    batch = features[0].shape[0]
    for index, (feature, expected_channels) in enumerate(zip(features, channels)):
        _check_bchw(feature, f"{module_name} input {index}")
        if feature.shape[0] != batch:
            raise ValueError(f"{module_name} inputs must share one batch size.")
        if feature.shape[1] != expected_channels:
            raise ValueError(
                f"{module_name} input {index} expected {expected_channels} channels, "
                f"got {feature.shape[1]}."
            )


def _resize(x: Tensor, size: tuple[int, int]) -> Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ConvBNAct(nn.Sequential):
    """Conv2d -> BatchNorm2d -> SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        if min(c1, c2, kernel_size, stride, groups) <= 0:
            raise ValueError("Convolution arguments must be positive.")
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                c1,
                c2,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )


@dataclass(frozen=True)
class WindowMeta:
    """Geometry required to reverse a padded window partition."""

    batch: int
    channels: int
    height: int
    width: int
    padded_height: int
    padded_width: int


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, WindowMeta]:
    """Pad BCHW on the bottom/right and return ``[B*nW, ws*ws, C]`` windows."""

    _check_bchw(x, "window_partition input")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    padded = F.pad(x, (0, pad_w, 0, pad_h))
    hp, wp = padded.shape[-2:]
    windows = (
        padded.reshape(
            b,
            c,
            hp // window_size,
            window_size,
            wp // window_size,
            window_size,
        )
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .reshape(-1, window_size * window_size, c)
    )
    return windows, WindowMeta(b, c, h, w, hp, wp)


def window_reverse(windows: Tensor, meta: WindowMeta, window_size: int) -> Tensor:
    """Reverse :func:`window_partition` and remove its bottom/right padding."""

    if windows.ndim != 3:
        raise ValueError(f"windows must be 3D, got shape {tuple(windows.shape)}.")
    expected_windows = (
        meta.batch
        * (meta.padded_height // window_size)
        * (meta.padded_width // window_size)
    )
    if windows.shape != (expected_windows, window_size * window_size, meta.channels):
        raise ValueError(
            "Window tensor does not match metadata: "
            f"got {tuple(windows.shape)}, expected "
            f"{(expected_windows, window_size * window_size, meta.channels)}."
        )
    x = (
        windows.reshape(
            meta.batch,
            meta.padded_height // window_size,
            meta.padded_width // window_size,
            window_size,
            window_size,
            meta.channels,
        )
        .permute(0, 5, 1, 3, 2, 4)
        .contiguous()
        .reshape(meta.batch, meta.channels, meta.padded_height, meta.padded_width)
    )
    return x[:, :, : meta.height, : meta.width].contiguous()


class WindowCrossAttention(nn.Module):
    """Windowed spatial self/cross-attention with relative position bias."""

    def __init__(
        self,
        q_channels: int,
        kv_channels: int,
        dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}.")
        self.q_channels = q_channels
        self.kv_channels = kv_channels
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5
        self.q = nn.Conv2d(q_channels, dim, 1, bias=True)
        self.kv = nn.Conv2d(kv_channels, 2 * dim, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

        relative_count = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_count, heads)
        )
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coordinates = coordinates.flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer(
            "relative_position_index",
            relative.sum(-1).to(torch.long),
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, q_feature: Tensor, kv_feature: Tensor | None = None) -> Tensor:
        kv_feature = q_feature if kv_feature is None else kv_feature
        _check_bchw(q_feature, "WindowCrossAttention query")
        _check_bchw(kv_feature, "WindowCrossAttention key/value")
        if q_feature.shape[0] != kv_feature.shape[0]:
            raise ValueError("Query and key/value batch sizes must match.")
        if q_feature.shape[1] != self.q_channels:
            raise ValueError(
                f"Expected {self.q_channels} query channels, got {q_feature.shape[1]}."
            )
        if kv_feature.shape[1] != self.kv_channels:
            raise ValueError(
                f"Expected {self.kv_channels} key/value channels, got {kv_feature.shape[1]}."
            )
        if q_feature.shape[-2:] != kv_feature.shape[-2:]:
            raise ValueError(
                "Window cross-attention requires aligned spatial sizes, got "
                f"{q_feature.shape[-2:]} and {kv_feature.shape[-2:]}."
            )

        q_windows, meta = window_partition(self.q(q_feature), self.window_size)
        kv_windows, kv_meta = window_partition(self.kv(kv_feature), self.window_size)
        if meta.height != kv_meta.height or meta.width != kv_meta.width:
            raise RuntimeError("Query and key/value partition metadata diverged.")
        k_windows, v_windows = kv_windows.chunk(2, dim=-1)
        tokens = self.window_size * self.window_size

        q = (
            q_windows.reshape(-1, tokens, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            k_windows.reshape(-1, tokens, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = (
            v_windows.reshape(-1, tokens, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_bias = (
            relative_bias.reshape(tokens, tokens, self.heads)
            .permute(2, 0, 1)
            .contiguous()
        )
        logits = logits + relative_bias.float().unsqueeze(0)
        valid = q_feature.new_ones(
            (q_feature.shape[0], 1, q_feature.shape[2], q_feature.shape[3])
        )
        valid_windows, _ = window_partition(valid, self.window_size)
        valid_keys = valid_windows[..., 0].to(torch.bool)
        logits = logits.masked_fill(
            ~valid_keys[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )
        logits = logits - logits.amax(dim=-1, keepdim=True)
        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, v.float()).to(q.dtype)
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(-1, tokens, self.dim)
        )
        output = window_reverse(output, meta, self.window_size)
        return self.proj(output)


class ChannelCrossAttention(nn.Module):
    """Cross-covariance channel self/cross-attention with positive temperature."""

    def __init__(
        self,
        q_channels: int,
        kv_channels: int,
        dim: int = GATHER_DIM,
        heads: int = HEADS,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}.")
        self.q_channels = q_channels
        self.kv_channels = kv_channels
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Sequential(
            nn.Conv2d(q_channels, dim, 1, bias=False),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
        )
        self.kv = nn.Sequential(
            nn.Conv2d(kv_channels, 2 * dim, 1, bias=False),
            nn.Conv2d(2 * dim, 2 * dim, 3, padding=1, groups=2 * dim, bias=False),
        )
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.log_temperature = nn.Parameter(torch.zeros(1, heads, 1, 1))

    @property
    def positive_temperature(self) -> Tensor:
        return F.softplus(self.log_temperature) + 1.0e-4

    def forward(self, q_feature: Tensor, kv_feature: Tensor | None = None) -> Tensor:
        kv_feature = q_feature if kv_feature is None else kv_feature
        _check_bchw(q_feature, "ChannelCrossAttention query")
        _check_bchw(kv_feature, "ChannelCrossAttention key/value")
        if q_feature.shape[0] != kv_feature.shape[0]:
            raise ValueError("Query and key/value batch sizes must match.")
        if q_feature.shape[1] != self.q_channels:
            raise ValueError(
                f"Expected {self.q_channels} query channels, got {q_feature.shape[1]}."
            )
        if kv_feature.shape[1] != self.kv_channels:
            raise ValueError(
                f"Expected {self.kv_channels} key/value channels, got {kv_feature.shape[1]}."
            )
        if q_feature.shape[-2:] != kv_feature.shape[-2:]:
            raise ValueError(
                "Channel cross-attention requires aligned spatial sizes, got "
                f"{q_feature.shape[-2:]} and {kv_feature.shape[-2:]}."
            )

        b, _, h, w = q_feature.shape
        spatial = h * w
        q = self.q(q_feature).reshape(b, self.heads, self.head_dim, spatial)
        kv = self.kv(kv_feature)
        k, v = kv.chunk(2, dim=1)
        k = k.reshape(b, self.heads, self.head_dim, spatial)
        v = v.reshape(b, self.heads, self.head_dim, spatial)

        q_normalized = F.normalize(q.float(), dim=-1, eps=1.0e-6)
        k_normalized = F.normalize(k.float(), dim=-1, eps=1.0e-6)
        logits = torch.matmul(q_normalized, k_normalized.transpose(-2, -1))
        logits = logits * self.positive_temperature.float()
        logits = logits - logits.amax(dim=-1, keepdim=True)
        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, v.float()).to(q.dtype)
        output = output.reshape(b, self.dim, h, w)
        return self.proj(output)


class AdaptiveInteraction(nn.Module):
    """Interact spatial and channel branches without directly adding them."""

    def __init__(self, dim: int = GATHER_DIM) -> None:
        super().__init__()
        hidden = max(dim // 4, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1, bias=True),
        )
        self.spatial_conv = nn.Conv2d(2, 1, 1, bias=True)
        self.fuse = nn.Conv2d(2 * dim, dim, 1, bias=True)

    def forward(self, spatial: Tensor, channel: Tensor) -> Tensor:
        if spatial.shape != channel.shape:
            raise ValueError(
                f"AdaptiveInteraction shapes differ: {spatial.shape} vs {channel.shape}."
            )
        channel_map = torch.sigmoid(
            self.channel_mlp(F.adaptive_avg_pool2d(channel, output_size=1))
        )
        spatial_summary = torch.cat(
            (
                spatial.mean(dim=1, keepdim=True),
                spatial.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_map = torch.sigmoid(self.spatial_conv(spatial_summary))
        return self.fuse(
            torch.cat((spatial * channel_map, channel * spatial_map), dim=1)
        )


class SGFN(nn.Module):
    """Spatial-gated feed-forward network with an internal residual."""

    def __init__(self, dim: int = GATHER_DIM, expansion: float = SGFN_EXPANSION) -> None:
        super().__init__()
        hidden = int(round(dim * expansion))
        if hidden <= 0:
            raise ValueError(f"Invalid SGFN hidden dimension: {hidden}.")
        self.in_proj = nn.Conv2d(dim, 2 * hidden, 1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            3,
            padding=1,
            groups=hidden,
            bias=True,
        )
        self.out_proj = nn.Conv2d(hidden, dim, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.in_proj(x).chunk(2, dim=1)
        gated = F.gelu(self.dwconv(x1)) * x2
        return x + self.out_proj(gated)


class DualAggregationBlock(nn.Module):
    """One spatial/channel aggregation, adaptive interaction, and SGFN block."""

    def __init__(
        self,
        dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
        sgfn_expansion: float = SGFN_EXPANSION,
    ) -> None:
        super().__init__()
        self.spatial = WindowCrossAttention(dim, dim, dim, heads, window_size)
        self.channel = ChannelCrossAttention(dim, dim, dim, heads)
        self.interaction = AdaptiveInteraction(dim)
        self.sgfn = SGFN(dim, sgfn_expansion)

    def forward(self, x: Tensor) -> Tensor:
        fused = self.interaction(self.spatial(x), self.channel(x))
        return self.sgfn(x + fused)


class SpatialSemanticCross(nn.Module):
    """Window cross-attention plus a local public-semantic branch and AIM."""

    def __init__(
        self,
        q_channels: int,
        kv_channels: int,
        dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()
        self.attention = WindowCrossAttention(
            q_channels,
            kv_channels,
            dim,
            heads,
            window_size,
        )
        self.local = nn.Sequential(
            nn.Conv2d(kv_channels, dim, 1, bias=False),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
        )
        self.interaction = AdaptiveInteraction(dim)

    def forward(self, query: Tensor, public_semantic: Tensor) -> Tensor:
        return self.interaction(
            self.attention(query, public_semantic),
            self.local(public_semantic),
        )


class ASCGDAlignC3(nn.Module):
    """C3 copy: depthwise stride-2 -> BN -> SiLU -> pointwise projection."""

    def __init__(self, c1: int, gather_dim: int = GATHER_DIM) -> None:
        super().__init__()
        self.c1 = c1
        self.gather_dim = gather_dim
        self.down = nn.Sequential(
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, gather_dim, 1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        _check_bchw(x, "ASCGDAlignC3 input")
        if x.shape[1] != self.c1:
            raise ValueError(f"ASCGDAlignC3 expected {self.c1} channels.")
        return self.down(x)


class ASCGDAlignC4(nn.Module):
    """C4 pointwise projection to the fixed gather width."""

    def __init__(self, c1: int, gather_dim: int = GATHER_DIM) -> None:
        super().__init__()
        self.c1 = c1
        self.gather_dim = gather_dim
        self.proj = ConvBNAct(c1, gather_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        _check_bchw(x, "ASCGDAlignC4 input")
        if x.shape[1] != self.c1:
            raise ValueError(f"ASCGDAlignC4 expected {self.c1} channels.")
        return self.proj(x)


class ASCGDAlignC5(nn.Module):
    """Project C5 at low resolution, then bilinearly upsample by two."""

    def __init__(self, c1: int, gather_dim: int = GATHER_DIM) -> None:
        super().__init__()
        self.c1 = c1
        self.gather_dim = gather_dim
        self.proj = ConvBNAct(c1, gather_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        _check_bchw(x, "ASCGDAlignC5 input")
        if x.shape[1] != self.c1:
            raise ValueError(f"ASCGDAlignC5 expected {self.c1} channels.")
        return F.interpolate(
            self.proj(x),
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )


class ASCGDGather(nn.Module):
    """Fuse A3/A4/A5 and return ``A4 + DualAggregation(G0)``."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
        sgfn_expansion: float = SGFN_EXPANSION,
        blocks: int = DUAL_AGGREGATION_BLOCKS,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("ASCGDGather requires channels for [A3, A4, A5].")
        if any(channel != gather_dim for channel in channels):
            raise ValueError(
                f"Gather inputs must all use gather_dim={gather_dim}, got {channels}."
            )
        if blocks != 1:
            raise ValueError("The scoped experiment fixes dual aggregation blocks to 1.")
        self.channels = tuple(channels)
        self.gather_dim = gather_dim
        self.heads = heads
        self.window_size = window_size
        self.sgfn_expansion = sgfn_expansion
        self.dual_aggregation_blocks = blocks
        self.fuse = ConvBNAct(3 * gather_dim, gather_dim, 1)
        self.blocks = nn.ModuleList(
            [
                DualAggregationBlock(
                    gather_dim,
                    heads,
                    window_size,
                    sgfn_expansion,
                )
            ]
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        _check_feature_list(features, self.channels, self.__class__.__name__)
        a3, a4, a5 = features
        size = a4.shape[-2:]
        g = self.fuse(torch.cat((_resize(a3, size), a4, _resize(a5, size)), dim=1))
        for block in self.blocks:
            g = block(g)
        return a4 + g


class _P3Base(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"{self.__class__.__name__} requires [C3, G] channels.")
        self.channels = tuple(channels)
        self.c3_channels, self.g_channels = self.channels

    def _inputs(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        _check_feature_list(features, self.channels, self.__class__.__name__)
        c3, g = features
        return c3, _resize(g, c3.shape[-2:])


class _P4Base(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(
                f"{self.__class__.__name__} requires [A3, A4, G] channels."
            )
        self.channels = tuple(channels)
        self.a3_channels, self.a4_channels, self.g_channels = self.channels

    def _inputs(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        _check_feature_list(features, self.channels, self.__class__.__name__)
        a3, a4, g = features
        size = a4.shape[-2:]
        return _resize(a3, size), a4, _resize(g, size)


class _P5Base(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int,
        *,
        build_down_p4: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(
                f"{self.__class__.__name__} requires [P4, C5, G] channels."
            )
        self.channels = tuple(channels)
        self.p4_channels, self.c5_channels, self.g_channels = self.channels
        self.gather_dim = gather_dim
        self.down_p4 = (
            nn.Sequential(
                nn.Conv2d(
                    self.p4_channels,
                    self.p4_channels,
                    3,
                    stride=2,
                    padding=1,
                    groups=self.p4_channels,
                    bias=False,
                ),
                nn.Conv2d(self.p4_channels, gather_dim, 1, bias=False),
            )
            if build_down_p4
            else None
        )

    def _inputs(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        _check_feature_list(features, self.channels, self.__class__.__name__)
        p4, c5, g = features
        down_p4 = (
            _resize(self.down_p4(p4), c5.shape[-2:])
            if self.down_p4 is not None
            else p4
        )
        return down_p4, c5, _resize(g, c5.shape[-2:])


class ASCGDDirectP3(_P3Base):
    """Direct center-to-P3 residual distribution."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels)
        self.proj = nn.Conv2d(self.g_channels, self.c3_channels, 1, bias=True)
        self.gamma3 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        c3, g = self._inputs(features)
        return c3 + self.gamma3 * self.proj(g)


class ASCGDDirectP4(_P4Base):
    """Direct center-to-P4 residual distribution anchored on A4."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels)
        self.proj = nn.Conv2d(self.g_channels, self.a4_channels, 1, bias=True)
        self.gamma4 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        _a3, a4, g = self._inputs(features)
        return a4 + self.gamma4 * self.proj(g)


class ASCGDDirectP5(_P5Base):
    """Direct center-to-P5 residual distribution."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels, gather_dim, build_down_p4=False)
        self.g_proj = nn.Conv2d(self.g_channels, self.c5_channels, 1, bias=True)
        self.gamma5 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        _down_p4, c5, g = self._inputs(features)
        return c5 + self.gamma5 * self.g_proj(g)


class ASCGDSpatialP3(_P3Base):
    """High-to-low window spatial cross-attention for P3."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels)
        self.cross = SpatialSemanticCross(
            self.c3_channels,
            self.g_channels,
            gather_dim,
            heads,
            window_size,
        )
        self.out_proj = nn.Conv2d(gather_dim, self.c3_channels, 1, bias=True)
        self.gamma3 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        c3, g = self._inputs(features)
        residual = self.out_proj(self.cross(c3, g))
        return c3 + self.gamma3 * residual


class ASCGDChannelP4(_P4Base):
    """Low-to-high channel cross-attention from A3 into P4."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        gamma_g: float = 0.1,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels)
        self.cross = ChannelCrossAttention(
            self.g_channels,
            self.a3_channels,
            gather_dim,
            heads,
        )
        self.g_proj = nn.Conv2d(self.g_channels, self.a4_channels, 1, bias=True)
        self.out_proj = nn.Conv2d(gather_dim, self.a4_channels, 1, bias=True)
        self.gamma_g = nn.Parameter(torch.tensor(float(gamma_g)))
        self.gamma4 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        a3, a4, g = self._inputs(features)
        residual = self.out_proj(self.cross(g, a3))
        return a4 + self.gamma_g * self.g_proj(g) + self.gamma4 * residual


class ASCGDChannelP5(_P5Base):
    """Low-to-high channel cross-attention from downsampled P4 into P5."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels, gather_dim)
        self.cross = ChannelCrossAttention(
            self.c5_channels,
            gather_dim,
            gather_dim,
            heads,
        )
        self.out_proj = nn.Conv2d(gather_dim, self.c5_channels, 1, bias=True)
        self.gamma5 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        down_p4, c5, _g = self._inputs(features)
        residual = self.out_proj(self.cross(c5, down_p4))
        return c5 + self.gamma5 * residual


class ASCGDSwappedP3(_P3Base):
    """Swapped high-to-low direction: channel cross-attention at P3."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels)
        self.cross = ChannelCrossAttention(
            self.c3_channels,
            self.g_channels,
            gather_dim,
            heads,
        )
        self.out_proj = nn.Conv2d(gather_dim, self.c3_channels, 1, bias=True)
        self.gamma3 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        c3, g = self._inputs(features)
        return c3 + self.gamma3 * self.out_proj(self.cross(c3, g))


class ASCGDSwappedP4(_P4Base):
    """Swapped low-to-high direction: spatial cross-attention at P4."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
        gamma_g: float = 0.1,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels)
        self.cross = SpatialSemanticCross(
            self.g_channels,
            self.a3_channels,
            gather_dim,
            heads,
            window_size,
        )
        self.g_proj = nn.Conv2d(self.g_channels, self.a4_channels, 1, bias=True)
        self.out_proj = nn.Conv2d(gather_dim, self.a4_channels, 1, bias=True)
        self.gamma_g = nn.Parameter(torch.tensor(float(gamma_g)))
        self.gamma4 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        a3, a4, g = self._inputs(features)
        return (
            a4
            + self.gamma_g * self.g_proj(g)
            + self.gamma4 * self.out_proj(self.cross(g, a3))
        )


class ASCGDSwappedP5(_P5Base):
    """Swapped low-to-high direction: spatial cross-attention at P5."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels, gather_dim)
        self.cross = SpatialSemanticCross(
            self.c5_channels,
            gather_dim,
            gather_dim,
            heads,
            window_size,
        )
        self.out_proj = nn.Conv2d(gather_dim, self.c5_channels, 1, bias=True)
        self.gamma5 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        down_p4, c5, _g = self._inputs(features)
        return c5 + self.gamma5 * self.out_proj(self.cross(c5, down_p4))


class ASCGDSymmetricP3(_P3Base):
    """Spatial and channel high-to-low cross-attention fused before P3 injection."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels)
        self.spatial = SpatialSemanticCross(
            self.c3_channels,
            self.g_channels,
            gather_dim,
            heads,
            window_size,
        )
        self.channel = ChannelCrossAttention(
            self.c3_channels,
            self.g_channels,
            gather_dim,
            heads,
        )
        self.fuse = nn.Conv2d(2 * gather_dim, self.c3_channels, 1, bias=True)
        self.gamma3 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        c3, g = self._inputs(features)
        residual = self.fuse(torch.cat((self.spatial(c3, g), self.channel(c3, g)), dim=1))
        return c3 + self.gamma3 * residual


class ASCGDSymmetricP4(_P4Base):
    """Spatial and channel low-to-high cross-attention fused for P4."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
        gamma_g: float = 0.1,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels)
        self.spatial = SpatialSemanticCross(
            self.g_channels,
            self.a3_channels,
            gather_dim,
            heads,
            window_size,
        )
        self.channel = ChannelCrossAttention(
            self.g_channels,
            self.a3_channels,
            gather_dim,
            heads,
        )
        self.fuse = nn.Conv2d(2 * gather_dim, self.a4_channels, 1, bias=True)
        self.g_proj = nn.Conv2d(self.g_channels, self.a4_channels, 1, bias=True)
        self.gamma_g = nn.Parameter(torch.tensor(float(gamma_g)))
        self.gamma4 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        a3, a4, g = self._inputs(features)
        residual = self.fuse(torch.cat((self.spatial(g, a3), self.channel(g, a3)), dim=1))
        return a4 + self.gamma_g * self.g_proj(g) + self.gamma4 * residual


class ASCGDSymmetricP5(_P5Base):
    """Spatial and channel low-to-high cross-attention fused for P5."""

    def __init__(
        self,
        channels: Sequence[int],
        gather_dim: int = GATHER_DIM,
        heads: int = HEADS,
        window_size: int = WINDOW_SIZE,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(channels, gather_dim)
        self.spatial = SpatialSemanticCross(
            self.c5_channels,
            gather_dim,
            gather_dim,
            heads,
            window_size,
        )
        self.channel = ChannelCrossAttention(
            self.c5_channels,
            gather_dim,
            gather_dim,
            heads,
        )
        self.fuse = nn.Conv2d(2 * gather_dim, self.c5_channels, 1, bias=True)
        self.gamma5 = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        down_p4, c5, _g = self._inputs(features)
        residual = self.fuse(
            torch.cat(
                (self.spatial(c5, down_p4), self.channel(c5, down_p4)),
                dim=1,
            )
        )
        return c5 + self.gamma5 * residual


ASCGD_DISTRIBUTION_MODULES = (
    ASCGDDirectP3,
    ASCGDDirectP4,
    ASCGDDirectP5,
    ASCGDSpatialP3,
    ASCGDChannelP4,
    ASCGDChannelP5,
    ASCGDSwappedP3,
    ASCGDSwappedP4,
    ASCGDSwappedP5,
    ASCGDSymmetricP3,
    ASCGDSymmetricP4,
    ASCGDSymmetricP5,
)


__all__ = [
    "ASCGDAlignC3",
    "ASCGDAlignC4",
    "ASCGDAlignC5",
    "ASCGDGather",
    "ASCGDDirectP3",
    "ASCGDDirectP4",
    "ASCGDDirectP5",
    "ASCGDSpatialP3",
    "ASCGDChannelP4",
    "ASCGDChannelP5",
    "ASCGDSwappedP3",
    "ASCGDSwappedP4",
    "ASCGDSwappedP5",
    "ASCGDSymmetricP3",
    "ASCGDSymmetricP4",
    "ASCGDSymmetricP5",
    "WindowCrossAttention",
    "ChannelCrossAttention",
    "DualAggregationBlock",
    "window_partition",
    "window_reverse",
]
