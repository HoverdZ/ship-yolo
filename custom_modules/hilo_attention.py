"""HiLo Attention adapted from the official LITv2 implementation for YOLO11 C2PSA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class HiLoAttention2d(nn.Module):
    """Official HiLo token attention with a safe NCHW adapter and dynamic padding."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 2,
        alpha: float = 0.5,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"HiLo dim {dim} must be divisible by num_heads {num_heads}.")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

        head_dim = dim // num_heads
        self.dim = dim
        self.window_size = int(window_size)
        self.low_heads = int(num_heads * alpha)
        self.low_dim = self.low_heads * head_dim
        self.high_heads = num_heads - self.low_heads
        self.high_dim = self.high_heads * head_dim
        if self.window_size == 1:
            self.high_heads = 0
            self.high_dim = 0
            self.low_heads = num_heads
            self.low_dim = dim
        self.scale = head_dim**-0.5

        if self.low_heads > 0:
            if self.window_size != 1:
                self.spatial_reduction = nn.AvgPool2d(self.window_size, self.window_size)
            self.low_q = nn.Linear(dim, self.low_dim, bias=qkv_bias)
            self.low_kv = nn.Linear(dim, self.low_dim * 2, bias=qkv_bias)
            self.low_projection = nn.Linear(self.low_dim, self.low_dim)

        if self.high_heads > 0:
            self.high_qkv = nn.Linear(dim, self.high_dim * 3, bias=qkv_bias)
            self.high_projection = nn.Linear(self.high_dim, self.high_dim)

    def _high_frequency(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        groups_h = height // self.window_size
        groups_w = width // self.window_size
        total_groups = groups_h * groups_w
        grouped = x.reshape(
            batch, groups_h, self.window_size, groups_w, self.window_size, self.dim
        ).transpose(2, 3)
        qkv = self.high_qkv(grouped).reshape(
            batch,
            total_groups,
            -1,
            3,
            self.high_heads,
            self.high_dim // self.high_heads,
        ).permute(3, 0, 1, 4, 2, 5)
        query, key, value = qkv.unbind(dim=0)
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        output = (attention @ value).transpose(2, 3).reshape(
            batch,
            groups_h,
            groups_w,
            self.window_size,
            self.window_size,
            self.high_dim,
        )
        output = output.transpose(2, 3).reshape(batch, height, width, self.high_dim)
        return self.high_projection(output)

    def _low_frequency(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        query = self.low_q(x).reshape(
            batch, height * width, self.low_heads, self.low_dim // self.low_heads
        ).permute(0, 2, 1, 3)
        if self.window_size > 1:
            reduced = self.spatial_reduction(x.permute(0, 3, 1, 2))
            reduced = reduced.reshape(batch, channels, -1).permute(0, 2, 1)
        else:
            reduced = x.reshape(batch, height * width, channels)
        key_value = self.low_kv(reduced).reshape(
            batch, -1, 2, self.low_heads, self.low_dim // self.low_heads
        ).permute(2, 0, 3, 1, 4)
        key, value = key_value.unbind(dim=0)
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(batch, height, width, self.low_dim)
        return self.low_projection(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.dim:
            raise ValueError(f"HiLo expected {self.dim} channels, got {channels}.")
        pad_h = (-height) % self.window_size
        pad_w = (-width) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[-2:]
        tokens = x.permute(0, 2, 3, 1)

        if self.high_heads == 0:
            output = self._low_frequency(tokens)
        elif self.low_heads == 0:
            output = self._high_frequency(tokens)
        else:
            output = torch.cat(
                (self._high_frequency(tokens), self._low_frequency(tokens)), dim=-1
            )
        output = output.permute(0, 3, 1, 2).contiguous()
        return output[:, :, :height, :width].reshape(batch, channels, height, width)


class _HiLoPSABlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        alpha: float,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.attention = HiLoAttention2d(
            channels,
            num_heads=num_heads,
            window_size=window_size,
            alpha=alpha,
        )
        self.ffn = nn.Sequential(
            Conv(channels, channels * 2, 1),
            Conv(channels * 2, channels, 1, act=False),
        )
        self.shortcut = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x) if self.shortcut else self.attention(x)
        return x + self.ffn(x) if self.shortcut else self.ffn(x)


class C2PSAHiLo(nn.Module):
    """C2PSA with only its self-attention interaction replaced by official HiLo."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        window_size: int = 2,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError(f"C2PSAHiLo requires c1 == c2, got {c1} and {c2}.")
        self.c = int(c1 * e)
        if self.c <= 0:
            raise ValueError(f"C2PSAHiLo hidden channels must be positive, got {self.c}.")
        num_heads = max(self.c // 64, 2)
        while num_heads > 1 and self.c % num_heads:
            num_heads -= 1
        if num_heads < 2:
            raise ValueError(
                f"C2PSAHiLo needs at least two divisible heads for Hi-Fi and Lo-Fi, got {self.c} channels."
            )

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(
                _HiLoPSABlock(self.c, num_heads, int(window_size), float(alpha))
                for _ in range(int(n))
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual, attended = self.cv1(x).split((self.c, self.c), dim=1)
        attended = self.m(attended)
        return self.cv2(torch.cat((residual, attended), dim=1))
