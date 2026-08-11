# SPDX-License-Identifier: AGPL-3.0-or-later
"""AC-YOLO modules adapted from the authors' official implementation.

Source: https://github.com/He-ship-sar/ACYOLO
Pinned source commit: 20dad8db5047add008e6eab65b032158f4a5d3e1

The repository source is based on Ultralytics 8.3.59.  This file preserves the
published ACmix/C2PSA_ACmix computations while using the installed Ultralytics
``Conv`` block and device-aware positional tensors for 8.4.92 compatibility.
The authors' CCFM is represented by the neck graph in the experiment YAML; it
is not a standalone layer in their released final model configuration.
"""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules import Conv


__all__ = ("ACmix", "C2PSA_ACmix")


def _position(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    """Return the official normalized 2-D positional grid on ``reference`` device."""

    loc_w = torch.linspace(
        -1.0,
        1.0,
        width,
        device=reference.device,
        dtype=reference.dtype,
    ).unsqueeze(0).repeat(height, 1)
    loc_h = torch.linspace(
        -1.0,
        1.0,
        height,
        device=reference.device,
        dtype=reference.dtype,
    ).unsqueeze(1).repeat(1, width)
    return torch.cat((loc_w.unsqueeze(0), loc_h.unsqueeze(0)), 0).unsqueeze(0)


class ACmix(nn.Module):
    """Official ACmix attention/convolution mixture used by AC-YOLO."""

    def __init__(
        self,
        in_planes: int,
        kernel_att: int = 7,
        head: int = 4,
        kernel_conv: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        head = max(int(head), 1)
        if in_planes % head:
            raise ValueError(
                f"ACmix channels ({in_planes}) must be divisible by heads ({head})."
            )
        self.in_planes = int(in_planes)
        self.out_planes = int(in_planes)
        self.head = head
        self.kernel_att = int(kernel_att)
        self.kernel_conv = int(kernel_conv)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.head_dim = self.out_planes // self.head

        self.rate1 = nn.Parameter(torch.tensor(0.5))
        self.rate2 = nn.Parameter(torch.tensor(0.5))
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.ReflectionPad2d(padding_att)
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_att,
            padding=0,
            stride=self.stride,
        )
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Conv2d(
            3 * self.head,
            self.kernel_conv * self.kernel_conv,
            kernel_size=1,
            bias=False,
        )
        self.dep_conv = nn.Conv2d(
            self.kernel_conv * self.kernel_conv * self.head_dim,
            self.out_planes,
            kernel_size=self.kernel_conv,
            bias=True,
            groups=self.head_dim,
            padding=self.kernel_conv // 2,
            stride=self.stride,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        kernel = torch.zeros(
            self.kernel_conv * self.kernel_conv,
            self.kernel_conv,
            self.kernel_conv,
        )
        for index in range(self.kernel_conv * self.kernel_conv):
            kernel[index, index // self.kernel_conv, index % self.kernel_conv] = 1.0
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        with torch.no_grad():
            self.dep_conv.weight.copy_(kernel)
            if self.dep_conv.bias is not None:
                self.dep_conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        batch, _channels, height, width = q.shape
        height_out = height // self.stride
        width_out = width // self.stride
        position = self.conv_p(_position(height, width, x))

        q_att = q.view(batch * self.head, self.head_dim, height, width)
        q_att = q_att * (float(self.head_dim) ** -0.5)
        k_att = k.view(batch * self.head, self.head_dim, height, width)
        v_att = v.view(batch * self.head, self.head_dim, height, width)
        if self.stride > 1:
            q_att = q_att[:, :, :: self.stride, :: self.stride]
            q_position = position[:, :, :: self.stride, :: self.stride]
        else:
            q_position = position

        unfold_k = self.unfold(self.pad_att(k_att)).view(
            batch * self.head,
            self.head_dim,
            self.kernel_att * self.kernel_att,
            height_out,
            width_out,
        )
        unfold_position = self.unfold(self.pad_att(position)).view(
            1,
            self.head_dim,
            self.kernel_att * self.kernel_att,
            height_out,
            width_out,
        )
        attention = (
            q_att.unsqueeze(2)
            * (unfold_k + q_position.unsqueeze(2) - unfold_position)
        ).sum(1)
        attention = self.softmax(attention)
        unfold_v = self.unfold(self.pad_att(v_att)).view(
            batch * self.head,
            self.head_dim,
            self.kernel_att * self.kernel_att,
            height_out,
            width_out,
        )
        out_attention = (attention.unsqueeze(1) * unfold_v).sum(2).view(
            batch,
            self.out_planes,
            height_out,
            width_out,
        )

        qkv = torch.cat(
            (
                q.view(batch, self.head, self.head_dim, height * width),
                k.view(batch, self.head, self.head_dim, height * width),
                v.view(batch, self.head, self.head_dim, height * width),
            ),
            dim=1,
        )
        convolution_weights = self.fc(qkv)
        convolution_features = convolution_weights.permute(0, 2, 1, 3).reshape(
            batch,
            -1,
            height,
            width,
        )
        out_convolution = self.dep_conv(convolution_features)
        return self.rate1 * out_attention + self.rate2 * out_convolution


class _ACmixPSABlock(nn.Module):
    """AC-YOLO's PSA residual block with ACmix replacing PSA attention."""

    def __init__(self, channels: int, num_heads: int, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = ACmix(channels, head=num_heads)
        self.ffn = nn.Sequential(
            Conv(channels, channels * 2, 1),
            Conv(channels * 2, channels, 1, act=False),
        )
        self.add = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA_ACmix(nn.Module):
    """Official C2PSA_ACmix block from the AC-YOLO backbone."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError(f"C2PSA_ACmix requires c1 == c2, got {c1} and {c2}.")
        hidden = int(c1 * e)
        self.c = hidden
        self.cv1 = Conv(c1, 2 * hidden, 1, 1)
        self.cv2 = Conv(2 * hidden, c1, 1)
        heads = max(hidden // 64, 1)
        self.m = nn.Sequential(
            *(_ACmixPSABlock(hidden, num_heads=heads) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), dim=1))
