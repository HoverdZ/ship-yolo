"""Ultralytics-compatible FDConv adapted from the official detection code.

The adaptation removes MMDetection registration, plotting, timing, and debug
utilities while retaining the method's FDW, global/local KSM, and FBM paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SafeBatchNorm2d(nn.BatchNorm2d):
    """Avoid a BatchNorm error only for YOLO's one-value construction pass."""

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


class StarReLU(nn.Module):
    """StarReLU used by the official FDConv global KSM."""

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.relu(x).square() + self.bias


def _fft2_frequency_indices(
    height: int, width: int, use_rfft: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return frequency coordinates ordered by radial frequency."""

    freq_h = torch.fft.fftfreq(int(height))
    freq_w = torch.fft.rfftfreq(int(width)) if use_rfft else torch.fft.fftfreq(int(width))
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing="ij"), dim=-1)
    distance = torch.linalg.vector_norm(freq_hw, dim=-1)
    indices = torch.argsort(distance.reshape(-1))
    frequency_width = int(width) // 2 + 1 if use_rfft else int(width)
    coordinates = torch.stack(
        (indices // frequency_width, indices % frequency_width), dim=-1
    )
    return coordinates.permute(1, 0), freq_hw


class KernelSpatialModulationGlobal(nn.Module):
    """Official global KSM channel/filter/spatial/kernel modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        reduction: float = 0.0625,
        kernel_num: int = 64,
        min_channel: int = 16,
        temperature: float = 1.0,
        kernel_temperature: float = 1.0,
        attention_multiplier: float = 2.0,
        channel_filter_act: str = "sigmoid",
        kernel_act: str = "sigmoid",
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        kernel_size = int(kernel_size)
        groups = int(groups)
        kernel_num = int(kernel_num)
        attention_channels = max(int(in_channels * float(reduction)), int(min_channel))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = float(temperature)
        self.kernel_temperature = float(kernel_temperature)
        self.attention_multiplier = float(attention_multiplier)
        self.channel_filter_act = channel_filter_act
        self.kernel_act = kernel_act

        self.fc = nn.Conv2d(in_channels, attention_channels, 1, bias=False)
        self.bn = _SafeBatchNorm2d(attention_channels)
        self.relu = StarReLU()
        self.channel_fc = nn.Conv2d(attention_channels, in_channels, 1, bias=True)
        self.filter_fc = None
        if not (in_channels == groups and in_channels == out_channels):
            self.filter_fc = nn.Conv2d(attention_channels, out_channels, 1, bias=True)
        self.spatial_fc = nn.Conv2d(
            attention_channels, kernel_size * kernel_size, 1, bias=True
        )
        self.kernel_fc = nn.Conv2d(attention_channels, kernel_num, 1, bias=True)
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
        nn.init.normal_(self.spatial_fc.weight, std=1e-6)
        nn.init.normal_(self.kernel_fc.weight, std=1e-6)
        nn.init.normal_(self.channel_fc.weight, std=1e-6)
        if self.filter_fc is not None:
            nn.init.normal_(self.filter_fc.weight, std=1e-6)

    def _channel_or_filter_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.channel_filter_act == "sigmoid":
            return torch.sigmoid(logits / self.temperature) * self.attention_multiplier
        if self.channel_filter_act == "tanh":
            return 1.0 + torch.tanh(logits / self.temperature)
        raise ValueError(f"Unsupported KSM activation: {self.channel_filter_act}.")

    def _kernel_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.kernel_act == "softmax":
            return F.softmax(logits / self.kernel_temperature, dim=1)
        if self.kernel_act == "sigmoid":
            return (
                torch.sigmoid(logits / self.kernel_temperature)
                * 2.0
                / float(self.kernel_num)
            )
        if self.kernel_act == "tanh":
            return (1.0 + torch.tanh(logits / self.kernel_temperature)) / float(
                self.kernel_num
            )
        raise ValueError(f"Unsupported KSM kernel activation: {self.kernel_act}.")

    def forward(
        self, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | float, torch.Tensor, torch.Tensor]:
        feature = self.relu(self.bn(self.fc(pooled)))
        batch = feature.shape[0]
        channel_attention = self._channel_or_filter_activation(
            self.channel_fc(feature)
        ).reshape(batch, 1, 1, self.in_channels, 1, 1)
        if self.filter_fc is None:
            filter_attention: torch.Tensor | float = 1.0
        else:
            filter_attention = self._channel_or_filter_activation(
                self.filter_fc(feature)
            ).reshape(batch, 1, self.out_channels, 1, 1, 1)
        spatial_attention = self._channel_or_filter_activation(
            self.spatial_fc(feature)
        ).reshape(batch, 1, 1, 1, self.kernel_size, self.kernel_size)
        kernel_logits = self.kernel_fc(feature).reshape(
            batch, self.kernel_num, 1, 1, 1, 1
        )
        kernel_attention = self._kernel_activation(kernel_logits)
        return channel_attention, filter_attention, spatial_attention, kernel_attention


class KernelSpatialModulationLocal(nn.Module):
    """Official dense element-wise local KSM implemented with Conv1d."""

    def __init__(self, channels: int, out_values: int, kernel_num: int = 1) -> None:
        super().__init__()
        channels = int(channels)
        out_values = int(out_values)
        kernel_num = int(kernel_num)
        conv_kernel = int(round((math.log2(channels) / 2.0) + 0.5) // 2 * 2 + 1)
        self.channels = channels
        self.kernel_num = kernel_num
        self.out_values = out_values
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(
            1,
            kernel_num * out_values,
            kernel_size=conv_kernel,
            padding=(conv_kernel - 1) // 2,
            bias=False,
        )
        nn.init.constant_(self.conv.weight, 1e-6)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        descriptor = pooled.squeeze(-1).transpose(-1, -2)
        descriptor = self.norm(descriptor)
        logits = self.conv(descriptor)
        logits = logits.reshape(
            pooled.shape[0], self.kernel_num, self.out_values, self.channels
        )
        return logits.permute(0, 1, 3, 2)


class FrequencyBandModulation(nn.Module):
    """FDConv FBM with official octave-band masks and spatial modulation."""

    def __init__(
        self,
        in_channels: int,
        k_list: tuple[int, ...] = (2, 4, 8),
        lowfreq_att: bool = False,
        act: str = "sigmoid",
        spatial_group: int = 1,
        spatial_kernel: int = 3,
        max_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        self.k_list = tuple(int(k) for k in k_list)
        self.lowfreq_att = bool(lowfreq_att)
        self.act = act
        self.spatial_group = int(spatial_group)
        if self.spatial_group > 64:
            self.spatial_group = in_channels
        if self.spatial_group <= 0 or in_channels % self.spatial_group:
            raise ValueError(
                f"FBM spatial_group={self.spatial_group} must divide channels={in_channels}."
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
            nn.init.normal_(conv.weight, std=1e-6)
            nn.init.zeros_(conv.bias)
            self.freq_weight_conv_list.append(conv)
        self.register_buffer(
            "cached_masks",
            self._precompute_masks(tuple(int(v) for v in max_size)),
            persistent=False,
        )

    def _precompute_masks(self, max_size: tuple[int, int]) -> torch.Tensor:
        _, frequencies = _fft2_frequency_indices(max_size[0], max_size[1], True)
        frequency_radius = frequencies.abs().amax(dim=-1)
        masks = [frequency_radius < 0.5 / divisor + 1e-8 for divisor in self.k_list]
        return torch.stack(masks, dim=0).unsqueeze(1)

    def _activate(self, weight: torch.Tensor) -> torch.Tensor:
        if self.act == "sigmoid":
            return weight.sigmoid() * 2.0
        if self.act == "tanh":
            return 1.0 + weight.tanh()
        if self.act == "softmax":
            return weight.softmax(dim=1) * weight.shape[1]
        raise ValueError(f"Unsupported FBM activation: {self.act}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        source = x.float()
        previous_low = source.clone()
        batch, _, height, width = source.shape
        spectrum = torch.fft.rfft2(source, norm="ortho")
        masks = F.interpolate(
            self.cached_masks.float(),
            size=(height, width // 2 + 1),
            mode="nearest",
        )
        selected: list[torch.Tensor] = []
        for index in range(len(self.k_list)):
            low_part = torch.fft.irfft2(
                spectrum * masks[index], s=(height, width), norm="ortho"
            )
            high_part = previous_low - low_part
            previous_low = low_part
            weight = self._activate(self.freq_weight_conv_list[index](x)).float()
            weighted = weight.reshape(
                batch, self.spatial_group, -1, height, width
            ) * high_part.reshape(batch, self.spatial_group, -1, height, width)
            selected.append(weighted.reshape(batch, -1, height, width))

        if self.lowfreq_att:
            weight = self._activate(
                self.freq_weight_conv_list[len(self.k_list)](x)
            ).float()
            weighted = weight.reshape(
                batch, self.spatial_group, -1, height, width
            ) * previous_low.reshape(batch, self.spatial_group, -1, height, width)
            selected.append(weighted.reshape(batch, -1, height, width))
        else:
            selected.append(previous_low)
        return torch.stack(selected, dim=0).sum(dim=0).to(original_dtype)


class FDConv(nn.Conv2d):
    """Frequency Dynamic Convolution retaining FDW, KSM, and FBM."""

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
        reduction: float = 0.0625,
        kernel_num: int = 64,
        kernel_temperature: float = 1.0,
        attention_multiplier: float = 2.0,
    ) -> None:
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        kernel_size = int(kernel_size)
        stride = int(stride)
        padding = int(padding)
        dilation = int(dilation)
        groups = int(groups)
        kernel_num = int(kernel_num)
        if min(in_channels, out_channels) <= 16:
            raise ValueError("FDConv requires both channel counts to be greater than 16.")
        if groups != 1:
            raise ValueError("This official detection-path adaptation requires groups=1.")
        if kernel_size != 3 or stride != 1 or dilation != 1:
            raise ValueError("The P2/P3 FDConv encoder requires 3x3 stride=1 dilation=1.")
        if kernel_num <= 1:
            raise ValueError("FDConv requires multiple disjoint frequency kernel groups.")

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bool(bias),
        )
        self.kernel_num = kernel_num
        self.param_ratio = 1
        self.attention_multiplier = float(attention_multiplier)
        self.alpha = (
            min(self.out_channels, self.in_channels) // 2 * self.kernel_num
        )
        self.KSM_Global = KernelSpatialModulationGlobal(
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            groups=self.groups,
            reduction=float(reduction),
            kernel_num=self.kernel_num,
            temperature=float(kernel_temperature),
            kernel_temperature=float(kernel_temperature),
            attention_multiplier=self.attention_multiplier,
            channel_filter_act="sigmoid",
            kernel_act="sigmoid",
        )
        self.KSM_Local = KernelSpatialModulationLocal(
            self.in_channels,
            self.out_channels * self.kernel_size[0] * self.kernel_size[1],
            kernel_num=1,
        )
        self.FBM = FrequencyBandModulation(
            self.in_channels,
            k_list=(2, 4, 8),
            lowfreq_att=False,
            act="sigmoid",
            spatial_group=1,
            spatial_kernel=3,
        )
        self._convert_to_fdw_parameter()

    def _convert_to_fdw_parameter(self) -> None:
        out_channels = self.out_channels
        in_channels = self.in_channels
        kernel_h, kernel_w = self.kernel_size
        indices, _ = _fft2_frequency_indices(
            out_channels * kernel_h, in_channels * kernel_w, True
        )
        coefficient_count = indices.shape[1]
        if coefficient_count % self.kernel_num:
            raise ValueError(
                f"FDW coefficient count {coefficient_count} is not divisible by "
                f"kernel_num={self.kernel_num}."
            )
        spatial_weight = self.weight.permute(0, 2, 1, 3).reshape(
            out_channels * kernel_h, in_channels * kernel_w
        )
        frequency_weight = torch.fft.rfft2(spatial_weight, dim=(0, 1))
        frequency_weight = torch.stack(
            (frequency_weight.real, frequency_weight.imag), dim=-1
        ).unsqueeze(0)
        frequency_weight = frequency_weight / float(
            min(self.out_channels, self.in_channels) // 2
        )
        self.dft_weight = nn.Parameter(frequency_weight, requires_grad=True)
        del self.weight
        grouped_indices = indices.reshape(2, self.kernel_num, -1)
        self.register_buffer(
            "indices", grouped_indices.unsqueeze(0), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"FDConv expected {self.in_channels} channels, got {x.shape[1]}.")
        batch, in_channels, height, width = x.shape
        pooled = F.adaptive_avg_pool2d(x, 1)
        channel_att, filter_att, spatial_att, kernel_att = self.KSM_Global(pooled)

        local_logits = self.KSM_Local(pooled).reshape(
            batch,
            1,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        local_logits = local_logits.permute(0, 1, 3, 2, 4, 5)
        local_att = local_logits.sigmoid() * self.attention_multiplier

        dft_height = self.out_channels * self.kernel_size[0]
        dft_width = self.in_channels * self.kernel_size[1] // 2 + 1
        dft_map = torch.zeros(
            (batch, dft_height, dft_width, 2),
            device=x.device,
            dtype=torch.float32,
        )
        kernel_att_fp32 = kernel_att.reshape(
            batch, self.param_ratio, self.kernel_num, -1
        ).float()
        indices = self.indices[0]
        weight_groups = (
            self.dft_weight.float()[0][indices[0], indices[1]].unsqueeze(0)
            * float(self.alpha)
        )
        values = torch.stack(
            (
                weight_groups[..., 0] * kernel_att_fp32[:, 0],
                weight_groups[..., 1] * kernel_att_fp32[:, 0],
            ),
            dim=-1,
        )
        dft_map[:, indices[0], indices[1]] += values
        adaptive_weight = torch.fft.irfft2(
            torch.view_as_complex(dft_map), dim=(1, 2)
        ).reshape(
            batch,
            1,
            self.out_channels,
            self.kernel_size[0],
            self.in_channels,
            self.kernel_size[1],
        )
        adaptive_weight = adaptive_weight.permute(0, 1, 2, 4, 3, 5)

        modulated_input = self.FBM(x)
        if isinstance(filter_att, float):
            filter_factor: torch.Tensor | float = filter_att
        else:
            filter_factor = filter_att
        aggregate_weight = (
            spatial_att.float()
            * channel_att.float()
            * (filter_factor if isinstance(filter_factor, float) else filter_factor.float())
            * adaptive_weight
            * local_att.float()
        ).sum(dim=1)
        aggregate_weight = aggregate_weight.reshape(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        ).to(modulated_input.dtype)
        grouped_input = modulated_input.reshape(1, batch * in_channels, height, width)
        output = F.conv2d(
            grouped_input,
            aggregate_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=batch * self.groups,
        )
        output = output.reshape(batch, self.out_channels, output.shape[-2], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output


class FDConvBlock(nn.Module):
    """FDConv plus the normalization/activation surrounding a detector conv."""

    def __init__(self, channels: int, kernel_num: int = 64) -> None:
        super().__init__()
        channels = int(channels)
        self.fdconv = FDConv(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            kernel_num=int(kernel_num),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.fdconv(x)))
