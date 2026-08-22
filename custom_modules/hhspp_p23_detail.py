"""HHSPP candidates that carry P2/P3 detail to the P5 context stage."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

from custom_modules.fadc_frequency import FreqSelectAdaKernConv
from custom_modules.fdconv import FDConvBlock
from custom_modules.hhspp import HHSPP


class _SafeBatchNorm2d(nn.BatchNorm2d):
    """Use running statistics for a one-value global-attention tensor."""

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


class SPDConv(nn.Module):
    """Official SPD block: space-to-depth factor 2, then stride-1 Conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"SPDConv requires positive channels, got {in_channels}->{out_channels}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = Conv(in_channels * 4, out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"SPDConv expected BCHW with {self.in_channels} channels, got {tuple(x.shape)}."
            )
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(
                f"SPDConv requires even spatial dimensions, got {height}x{width}."
            )
        # Preserve the slice order of the official SPD-Conv implementation.
        space_to_depth = torch.cat(
            (
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ),
            dim=1,
        )
        return self.conv(space_to_depth)


class _ASFFConv(nn.Sequential):
    """Conv-BN-LeakyReLU block used by the official ASFF code."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        padding = (int(kernel_size) - 1) // 2
        super().__init__(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                int(kernel_size),
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(int(out_channels)),
            nn.LeakyReLU(0.1, inplace=True),
        )


class ASFF2(nn.Module):
    """Two-input engineering adaptation of ASFF's spatial scale weighting."""

    def __init__(self, channels: int, compressed_channels: int = 16) -> None:
        super().__init__()
        channels = int(channels)
        compressed_channels = int(compressed_channels)
        if channels <= 0 or compressed_channels <= 0:
            raise ValueError("ASFF2 channels must be positive integers.")
        self.channels = channels
        self.weight_feature_0 = _ASFFConv(channels, compressed_channels, 1)
        self.weight_feature_1 = _ASFFConv(channels, compressed_channels, 1)
        self.weight_logits = nn.Conv2d(compressed_channels * 2, 2, 1, bias=True)
        self.expand = _ASFFConv(channels, channels, 3)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        if x0.shape != x1.shape:
            raise ValueError(
                f"ASFF2 requires aligned inputs, got {tuple(x0.shape)} and {tuple(x1.shape)}."
            )
        if x0.shape[1] != self.channels:
            raise ValueError(f"ASFF2 expected {self.channels} channels, got {x0.shape[1]}.")
        logits = self.weight_logits(
            torch.cat((self.weight_feature_0(x0), self.weight_feature_1(x1)), dim=1)
        )
        weights = F.softmax(logits, dim=1)
        fused = x0 * weights[:, 0:1] + x1 * weights[:, 1:2]
        return self.expand(fused)


def _attention_path(channels: int, hidden_channels: int, global_pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    if global_pool:
        layers.append(nn.AdaptiveAvgPool2d(1))
    layers.extend(
        (
            nn.Conv2d(channels, hidden_channels, 1, bias=True),
            _SafeBatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
            _SafeBatchNorm2d(channels),
        )
    )
    return nn.Sequential(*layers)


class IterativeAFF(nn.Module):
    """Two-stage iAFF with independent local/global parameters per stage."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        channels = int(channels)
        reduction = int(reduction)
        if channels <= 0 or reduction <= 0:
            raise ValueError("iAFF channels and reduction must be positive integers.")
        hidden_channels = max(1, channels // reduction)
        self.channels = channels
        self.local_att = _attention_path(channels, hidden_channels, global_pool=False)
        self.global_att = _attention_path(channels, hidden_channels, global_pool=True)
        self.local_att2 = _attention_path(channels, hidden_channels, global_pool=False)
        self.global_att2 = _attention_path(channels, hidden_channels, global_pool=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, context: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        if context.shape != detail.shape:
            raise ValueError(
                "iAFF requires context/detail shape equality, got "
                f"{tuple(context.shape)} and {tuple(detail.shape)}."
            )
        if context.shape[1] != self.channels:
            raise ValueError(
                f"iAFF expected {self.channels} channels, got {context.shape[1]}."
            )
        combined = context + detail
        weight = self.sigmoid(self.local_att(combined) + self.global_att(combined))
        intermediate = context * weight + detail * (1.0 - weight)

        # The second stage deliberately uses global_att2.  Some PyTorch copies
        # of the reference repository accidentally call global_att here again.
        weight2 = self.sigmoid(
            self.local_att2(intermediate) + self.global_att2(intermediate)
        )
        return context * weight2 + detail * (1.0 - weight2)


class HWDDown(nn.Module):
    """Official HWD: Haar DWT four-band concat followed by 1x1 Conv-BN-ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        try:
            from pytorch_wavelets import DWTForward
        except ImportError as error:
            raise ImportError(
                "HHSPPP23HWD requires the official dependency: "
                "pip install pytorch-wavelets==1.3.0"
            ) from error
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet = DWTForward(J=1, mode="zero", wave="haar")
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"HWD expected BCHW with {self.in_channels} channels, got {tuple(x.shape)}."
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                f"HWD detail alignment requires even H/W, got {tuple(x.shape[-2:])}."
            )
        low_low, high_bands = self.wavelet(x)
        high_low = high_bands[0][:, :, 0, ...]
        low_high = high_bands[0][:, :, 1, ...]
        high_high = high_bands[0][:, :, 2, ...]
        four_bands = torch.cat((low_low, high_low, low_high, high_high), dim=1)
        if four_bands.shape[1] != self.in_channels * 4:
            raise RuntimeError("HWD did not produce LL/HL/LH/HH for every input channel.")
        return self.projection(four_bands)


def _normalize_channels(
    in_channels: Sequence[int], output_channels: int
) -> tuple[int, int, int, int]:
    if isinstance(in_channels, (str, bytes)) or len(in_channels) != 3:
        raise ValueError("HHSPP P2/P3 detail modules require exactly [P2, P3, P5] channels.")
    channels = tuple(int(channel) for channel in in_channels)
    if any(channel <= 0 for channel in channels):
        raise ValueError(f"Input channels must be positive, got {channels}.")
    output_channels = int(output_channels)
    if output_channels <= 0:
        raise ValueError(f"Output channels must be positive, got {output_channels}.")
    return channels[0], channels[1], channels[2], output_channels


class _HHSPPP23Base(HHSPP):
    """Common direct HHSPP inheritance that preserves cv1/cv2 state keys."""

    def __init__(self, in_channels: Sequence[int], c2: int) -> None:
        c_p2, c_p3, c_p5, c2 = _normalize_channels(in_channels, c2)
        super().__init__(c_p5, c2)
        self.input_channels = (c_p2, c_p3, c_p5)
        self.output_channels = c2

    def _unpack(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise ValueError("HHSPP P2/P3 detail modules require [p2, p3, p5] tensors.")
        p2, p3, p5 = features
        for tensor, expected_channels, name in zip(
            (p2, p3, p5), self.input_channels, ("P2", "P3", "P5"), strict=True
        ):
            if tensor.ndim != 4 or tensor.shape[1] != expected_channels:
                raise ValueError(
                    f"{name} expected BCHW/{expected_channels} channels, got {tuple(tensor.shape)}."
                )
        return p2, p3, p5

    def _context(self, p5: torch.Tensor) -> torch.Tensor:
        return HHSPP.forward(self, p5)


class HHSPPP23Concat(_HHSPPP23Base):
    """Minimal P2/P3 bypass: standard Conv/Concat only."""

    def __init__(self, in_channels: Sequence[int], c2: int) -> None:
        super().__init__(in_channels, c2)
        c_p2, c_p3, _ = self.input_channels
        c2 = self.output_channels
        self.p2_down = Conv(c_p2, c_p3, 3, 2)
        self.p3_fuse = Conv(c_p3 * 2, c_p3, 1, 1)
        self.detail_down_p4 = Conv(c_p3, c2, 3, 2)
        self.detail_down_p5 = Conv(c2, c2, 3, 2)
        self.final_fusion = Conv(c2 * 2, c2, 1, 1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        p2, p3, p5 = self._unpack(features)
        p2_at_p3 = self.p2_down(p2)
        if p2_at_p3.shape[-2:] != p3.shape[-2:]:
            raise ValueError("P2 stride-2 alignment does not match P3 resolution.")
        detail_p3 = self.p3_fuse(torch.cat((p2_at_p3, p3), dim=1))
        detail_p5 = self.detail_down_p5(self.detail_down_p4(detail_p3))
        context = self._context(p5)
        if context.shape != detail_p5.shape:
            raise ValueError("Concat candidate context/detail shapes are not aligned at P5.")
        return self.final_fusion(torch.cat((context, detail_p5), dim=1))


class _HHSPPP23AdaptiveBase(_HHSPPP23Base):
    """Common ASFF2 + two-downsample + iAFF execution path."""

    def _initialize_fusion(
        self,
        p2_to_p3: nn.Module,
        p3_encoder: nn.Module,
        p3_to_p4: nn.Module,
        p4_to_p5: nn.Module,
    ) -> None:
        detail_channels = self.input_channels[1]
        self.p2_to_p3 = p2_to_p3
        self.p3_encoder = p3_encoder
        self.asff2 = ASFF2(detail_channels)
        self.detail_down_p4 = p3_to_p4
        self.detail_down_p5 = p4_to_p5
        self.iaff = IterativeAFF(self.output_channels, reduction=4)

    def _forward_adaptive(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        p2, p3, p5 = self._unpack(features)
        p2_at_p3 = self.p2_to_p3(p2)
        p3_encoded = self.p3_encoder(p3)
        detail_p3 = self.asff2(p2_at_p3, p3_encoded)
        detail_p5 = self.detail_down_p5(self.detail_down_p4(detail_p3))
        context = self._context(p5)
        if context.shape != detail_p5.shape:
            raise ValueError("Adaptive candidate context/detail shapes are not aligned at P5.")
        return self.iaff(context, detail_p5)


class HHSPPP23SPD(_HHSPPP23AdaptiveBase):
    """P2/P3 detail path using three official SPD downsampling blocks."""

    def __init__(self, in_channels: Sequence[int], c2: int) -> None:
        super().__init__(in_channels, c2)
        c_p2, c_p3, _ = self.input_channels
        c2 = self.output_channels
        self._initialize_fusion(
            SPDConv(c_p2, c_p3),
            Conv(c_p3, c_p3, 1, 1),
            SPDConv(c_p3, c2),
            SPDConv(c2, c2),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._forward_adaptive(features)


class HHSPPP23FreqAdaKern(_HHSPPP23AdaptiveBase):
    """P2/P3 FrequencySelection + AdaKern, explicitly excluding AdaDR/DCN."""

    def __init__(self, in_channels: Sequence[int], c2: int) -> None:
        super().__init__(in_channels, c2)
        c_p2, c_p3, _ = self.input_channels
        c2 = self.output_channels
        p2_frequency_adakern = FreqSelectAdaKernConv(c_p2)
        p3_frequency_adakern = FreqSelectAdaKernConv(c_p3)
        self._initialize_fusion(
            nn.Sequential(p2_frequency_adakern, SPDConv(c_p2, c_p3)),
            nn.Sequential(p3_frequency_adakern, Conv(c_p3, c_p3, 1, 1)),
            SPDConv(c_p3, c2),
            SPDConv(c2, c2),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._forward_adaptive(features)


class HHSPPP23FDConv(_HHSPPP23AdaptiveBase):
    """P2/P3 FDConv encoders retaining FDW, KSM global/local, and FBM."""

    def __init__(self, in_channels: Sequence[int], c2: int) -> None:
        super().__init__(in_channels, c2)
        c_p2, c_p3, _ = self.input_channels
        c2 = self.output_channels
        p2_fdconv = FDConvBlock(c_p2, kernel_num=64)
        p3_fdconv = FDConvBlock(c_p3, kernel_num=64)
        self._initialize_fusion(
            nn.Sequential(p2_fdconv, SPDConv(c_p2, c_p3)),
            nn.Sequential(p3_fdconv, Conv(c_p3, c_p3, 1, 1)),
            SPDConv(c_p3, c2),
            SPDConv(c2, c2),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._forward_adaptive(features)


class HHSPPP23HWD(_HHSPPP23AdaptiveBase):
    """P2/P3 detail path using official four-subband Haar downsampling."""

    def __init__(self, in_channels: Sequence[int], c2: int) -> None:
        super().__init__(in_channels, c2)
        c_p2, c_p3, _ = self.input_channels
        c2 = self.output_channels
        self._initialize_fusion(
            HWDDown(c_p2, c_p3),
            Conv(c_p3, c_p3, 1, 1),
            HWDDown(c_p3, c2),
            HWDDown(c2, c2),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._forward_adaptive(features)


P23_HHSPP_MODULES = frozenset(
    {
        HHSPPP23Concat,
        HHSPPP23SPD,
        HHSPPP23FreqAdaKern,
        HHSPPP23FDConv,
        HHSPPP23HWD,
    }
)
