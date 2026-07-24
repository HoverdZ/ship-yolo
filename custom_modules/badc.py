"""Background-aware directional contrast blocks for shallow ship features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules import C2f, Conv


class BackgroundAwareDirectionalContrast(nn.Module):
    """Extract local contrast with spatially selected directional depthwise filters.

    A local average is treated as a low-frequency background estimate. Three
    full-channel depthwise branches operate on the residual contrast, while a
    cheap two-channel spatial gate selects local, horizontal, or vertical
    evidence at every position. A zero-initialized layer scale makes the block
    start as a no-op inside the surrounding YOLO residual bottleneck.
    """

    def __init__(
        self,
        channels: int,
        background_kernel: int = 7,
        band_kernel: int = 11,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if background_kernel <= 0 or background_kernel % 2 == 0:
            raise ValueError("background_kernel must be a positive odd integer.")
        if band_kernel <= 0 or band_kernel % 2 == 0:
            raise ValueError("band_kernel must be a positive odd integer.")

        self.channels = int(channels)
        self.background_kernel = int(background_kernel)
        self.band_kernel = int(band_kernel)
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            (1, band_kernel),
            padding=(0, band_kernel // 2),
            groups=channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            (band_kernel, 1),
            padding=(band_kernel // 2, 0),
            groups=channels,
            bias=False,
        )
        self.spatial_gate = nn.Conv2d(2, 3, 3, padding=1, bias=True)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        self.layer_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))

        # Equal branch selection at initialization. The transformed residual is
        # still exactly zero when layer_scale_init=0.
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return low-frequency background and residual local contrast."""

        background = F.avg_pool2d(
            x,
            kernel_size=self.background_kernel,
            stride=1,
            padding=self.background_kernel // 2,
            count_include_pad=False,
        )
        return background, x - background

    def branch_weights(
        self, background: torch.Tensor, contrast: torch.Tensor
    ) -> torch.Tensor:
        """Produce per-position local/horizontal/vertical softmax weights."""

        descriptors = torch.cat(
            (
                background.abs().mean(dim=1, keepdim=True),
                contrast.abs().mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        return self.spatial_gate(descriptors).softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW with {self.channels} channels, got {tuple(x.shape)}."
            )
        background, contrast = self.decompose(x)
        weights = self.branch_weights(background, contrast)
        branches = torch.stack(
            (
                self.local(contrast),
                self.horizontal(contrast),
                self.vertical(contrast),
            ),
            dim=1,
        )
        mixed = (branches * weights.unsqueeze(2)).sum(dim=1)
        return self.layer_scale * self.act(self.norm(mixed))


class BADCBottleneck(nn.Module):
    """YOLO11 bottleneck retaining cv1 and replacing only the second spatial op."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        if g != 1:
            raise ValueError("BADC shallow experiments require g=1.")
        hidden_channels = int(c2 * e)
        if hidden_channels <= 0:
            raise ValueError(f"Invalid hidden channels from c2={c2}, e={e}.")
        self.cv1 = Conv(c1, hidden_channels, k[0], 1)
        self.cv2_adapter: nn.Module = (
            nn.Conv2d(hidden_channels, c2, 1, bias=False)
            if hidden_channels != c2
            else nn.Identity()
        )
        self.cv2 = BackgroundAwareDirectionalContrast(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv2_adapter(self.cv1(x)))
        return x + y if self.add else y


class C3k2_BADC(C2f):
    """C3k2-compatible BADC module limited to non-C3k shallow blocks."""

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
        if c3k:
            raise ValueError("C3k2_BADC is limited to c3k=False P2/P3 blocks.")
        if attn:
            raise ValueError("C3k2_BADC does not add channel attention.")
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(
            BADCBottleneck(self.c, self.c, shortcut=shortcut, g=g)
            for _ in range(n)
        )
