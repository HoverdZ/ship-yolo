"""FrequencySelection + AdaKern extracted from the official FADC implementation.

This file intentionally implements only the frequency-selection and adaptive-
kernel parts requested by the controlled experiment.  It contains no AdaDR,
deformable offsets, deformable masks, adaptive dilation, or deformable
convolution dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SafeBatchNorm2d(nn.BatchNorm2d):
    """Use running statistics for a degenerate one-value construction batch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values_per_channel = x.numel() // x.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(x)


class FrequencySelection(nn.Module):
    """Official FADC frequency-band selection with the controlled config."""

    def __init__(
        self,
        in_channels: int,
        k_list: tuple[int, ...] = (2, 4, 8),
        lowfreq_att: bool = False,
        lp_type: str = "freq",
        act: str = "sigmoid",
        spatial: str = "conv",
        spatial_group: int = 1,
        spatial_kernel: int = 3,
        global_selection: bool = False,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        spatial_group = int(spatial_group)
        self.k_list = tuple(int(k) for k in k_list)
        self.lowfreq_att = bool(lowfreq_att)
        self.lp_type = lp_type
        self.act = act
        self.spatial_group = in_channels if spatial_group > 64 else spatial_group
        self.global_selection = bool(global_selection)

        if in_channels <= 0:
            raise ValueError(f"FrequencySelection requires positive channels, got {in_channels}.")
        if self.lp_type != "freq":
            raise ValueError("This controlled FADC-lite experiment requires lp_type='freq'.")
        if spatial != "conv":
            raise ValueError("FrequencySelection currently preserves the official spatial='conv' path only.")
        if self.act != "sigmoid":
            raise ValueError("This controlled experiment requires act='sigmoid'.")
        if self.global_selection:
            raise ValueError("This controlled experiment requires global_selection=False.")
        if not self.k_list or any(k <= 0 for k in self.k_list):
            raise ValueError(f"Invalid frequency divisors: {self.k_list}.")
        if self.spatial_group <= 0 or in_channels % self.spatial_group:
            raise ValueError(
                f"spatial_group={self.spatial_group} must divide in_channels={in_channels}."
            )

        branch_count = len(self.k_list) + int(self.lowfreq_att)
        self.freq_weight_conv_list = nn.ModuleList()
        for _ in range(branch_count):
            conv = nn.Conv2d(
                in_channels,
                self.spatial_group,
                kernel_size=int(spatial_kernel),
                stride=1,
                padding=int(spatial_kernel) // 2,
                groups=self.spatial_group,
                bias=True,
            )
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            self.freq_weight_conv_list.append(conv)

    @staticmethod
    def _spatial_activation(weight: torch.Tensor) -> torch.Tensor:
        return weight.sigmoid() * 2.0

    def forward(self, x: torch.Tensor, att_feat: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FrequencySelection expects BCHW input, got shape {tuple(x.shape)}.")
        if att_feat is None:
            att_feat = x

        original_dtype = x.dtype
        source = x.float()
        previous_low = source.clone()
        batch, _, height, width = source.shape
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(source, norm="ortho"), dim=(-2, -1)
        )
        selected: list[torch.Tensor] = []

        # The rectangular masks depend on the current tensor shape, so they
        # cannot be fixed buffers without hard-coding an input resolution.
        for index, divisor in enumerate(self.k_list):
            mask = source.new_zeros((1, 1, height, width))
            h0 = round(height / 2 - height / (2 * divisor))
            h1 = round(height / 2 + height / (2 * divisor))
            w0 = round(width / 2 - width / (2 * divisor))
            w1 = round(width / 2 + width / (2 * divisor))
            mask[:, :, h0:h1, w0:w1] = 1.0
            low_part = torch.fft.ifft2(
                torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), norm="ortho"
            ).real
            high_part = previous_low - low_part
            previous_low = low_part

            frequency_weight = self._spatial_activation(
                self.freq_weight_conv_list[index](att_feat)
            ).float()
            weighted = frequency_weight.reshape(
                batch, self.spatial_group, -1, height, width
            ) * high_part.reshape(batch, self.spatial_group, -1, height, width)
            selected.append(weighted.reshape(batch, -1, height, width))

        if self.lowfreq_att:
            frequency_weight = self._spatial_activation(
                self.freq_weight_conv_list[len(self.k_list)](att_feat)
            ).float()
            weighted = frequency_weight.reshape(
                batch, self.spatial_group, -1, height, width
            ) * previous_low.reshape(batch, self.spatial_group, -1, height, width)
            selected.append(weighted.reshape(batch, -1, height, width))
        else:
            selected.append(previous_low)
        return torch.stack(selected, dim=0).sum(dim=0).to(original_dtype)


class OmniAttention(nn.Module):
    """FADC OmniAttention used by the low/high AdaKern decomposition."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        groups: int = 1,
        reduction: float = 0.0625,
        min_channel: int = 16,
    ) -> None:
        super().__init__()
        in_planes = int(in_planes)
        out_planes = int(out_planes)
        groups = int(groups)
        attention_channels = max(int(in_planes * float(reduction)), int(min_channel))
        self.temperature = 1.0
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channels, 1, bias=False)
        self.bn = _SafeBatchNorm2d(attention_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc = nn.Conv2d(attention_channels, in_planes, 1, bias=True)
        self.filter_fc = None
        if not (in_planes == groups and in_planes == out_planes):
            self.filter_fc = nn.Conv2d(attention_channels, out_planes, 1, bias=True)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | float]:
        pooled = self.relu(self.bn(self.fc(self.avgpool(x))))
        channel_attention = torch.sigmoid(self.channel_fc(pooled) / self.temperature)
        if self.filter_fc is None:
            filter_attention: torch.Tensor | float = 1.0
        else:
            filter_attention = torch.sigmoid(self.filter_fc(pooled) / self.temperature)
        return channel_attention, filter_attention


class AdaKernConv2d(nn.Module):
    """Ordinary dynamic Conv2d using FADC's mean/residual AdaKern weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)
        if self.groups != 1:
            raise ValueError("The controlled AdaKern branch currently requires groups=1.")
        if self.kernel_size != 3 or self.stride != 1 or self.dilation != 1:
            raise ValueError("FreqAdaKern requires ordinary 3x3, stride=1, dilation=1 Conv2d.")

        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bool(bias) else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # The official kernel_decompose="both" path uses two independent
        # OmniAttention blocks for kernel mean and residual components.
        self.OMNI_ATT1 = OmniAttention(self.in_channels, self.out_channels, groups=1)
        self.OMNI_ATT2 = OmniAttention(self.in_channels, self.out_channels, groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"AdaKern expected {self.in_channels} channels, got {x.shape[1]}."
            )
        channel_low, filter_low = self.OMNI_ATT1(x)
        channel_high, filter_high = self.OMNI_ATT2(x)
        if isinstance(filter_low, float) or isinstance(filter_high, float):
            raise RuntimeError("AdaKern's non-depthwise branch requires learned filter attention.")

        batch = x.shape[0]
        base_weight = self.weight.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        weight_mean = base_weight.mean(dim=(-1, -2), keepdim=True)
        weight_residual = base_weight - weight_mean
        adaptive_weight = (
            weight_mean
            * (channel_low.unsqueeze(1) * 2.0)
            * (filter_low.unsqueeze(2) * 2.0)
            + weight_residual
            * (channel_high.unsqueeze(1) * 2.0)
            * (filter_high.unsqueeze(2) * 2.0)
        )
        adaptive_weight = adaptive_weight.reshape(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        grouped_input = x.reshape(1, batch * self.in_channels, x.shape[-2], x.shape[-1])
        grouped_bias = self.bias.repeat(batch) if self.bias is not None else None
        output = F.conv2d(
            grouped_input,
            adaptive_weight,
            grouped_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=batch,
        )
        return output.reshape(batch, self.out_channels, output.shape[-2], output.shape[-1])


class FreqSelectAdaKernConv(nn.Module):
    """FrequencySelection followed by non-deformable AdaKern and YOLO norm/act."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.frequency_selection = FrequencySelection(
            channels,
            k_list=(2, 4, 8),
            lowfreq_att=False,
            lp_type="freq",
            act="sigmoid",
            spatial="conv",
            spatial_group=1,
            spatial_kernel=3,
            global_selection=False,
        )
        self.adakern = AdaKernConv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.adakern(self.frequency_selection(x))))
