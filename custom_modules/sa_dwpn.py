"""SA-DWPN modules for YOLO11 remote sensing ship detection experiments.

SA-DWPN-B keeps the standard three detection heads and injects C2 detail into
P3 through a downsampled guidance path. The dynamic fusion block is intentionally
small and stable for first-stage SCI ablation experiments.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules import Conv
except Exception:  # pragma: no cover - used only outside an Ultralytics source tree
    Conv = None


class _FallbackConv(nn.Module):
    """Small Conv-BN-SiLU fallback used when Ultralytics Conv is unavailable."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def _conv(c1: int, c2: int, k: int = 1, s: int = 1) -> nn.Module:
    """Create an Ultralytics Conv when available, otherwise use a local fallback."""

    if Conv is not None:
        return Conv(c1, c2, k, s)
    return _FallbackConv(c1, c2, k, s)


class Align(nn.Module):
    """Channel alignment block for feature pyramid inputs."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1) -> None:
        super().__init__()
        self.cv = _conv(c1, c2, k, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv(x)


class DWDown(nn.Module):
    """Downsample C2 detail features from stride 4 to stride 8."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.cv = _conv(c1, c2, k=3, s=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv(x)


class SDWF(nn.Module):
    """Scene-Adaptive Dynamic Weighted Fusion.

    The first-stage block combines static learnable weights, optional image-level
    dynamic weights, and an output Conv. Spatial gating is implemented but kept
    disabled by default for the SA-DWPN-B experiment.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n_inputs: int = 2,
        use_dynamic: bool = True,
        use_spatial: bool = False,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if n_inputs < 1:
            raise ValueError(f"SDWF requires at least one input, got {n_inputs}")

        self.c1 = c1
        self.c2 = c2
        self.n_inputs = n_inputs
        self.use_dynamic = use_dynamic
        self.use_spatial = use_spatial
        self.eps = eps

        self.static_w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        if use_dynamic:
            hidden = max(c1 // 4, 16)
            self.gate = nn.Sequential(
                nn.Linear(c1 * n_inputs, hidden),
                nn.SiLU(),
                nn.Linear(hidden, n_inputs),
            )
        else:
            self.gate = None

        if use_spatial:
            self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=True)
            self.eta = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.spatial_gate = None
            self.eta = None

        self.out_conv = _conv(c1, c2, k=3, s=1)

    def _as_list(self, x: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(x, torch.Tensor):
            x = [x]
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"SDWF expects a Tensor or list/tuple of Tensors, got {type(x).__name__}")
        if len(x) != self.n_inputs:
            raise ValueError(f"SDWF expects {self.n_inputs} inputs, got {len(x)}")
        return list(x)

    def _align_size(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        target_size = xs[0].shape[-2:]
        return [
            F.interpolate(t, size=target_size, mode="nearest") if t.shape[-2:] != target_size else t
            for t in xs
        ]

    def _check_channels(self, xs: list[torch.Tensor]) -> None:
        channels = [t.shape[1] for t in xs]
        if any(c != self.c1 for c in channels):
            raise ValueError(
                "SDWF requires channel-aligned inputs. "
                f"Expected all channels={self.c1}, got {channels}. "
                "Use Conv/Align layers in YAML before SDWF."
            )

    def _apply_spatial_gate(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.use_spatial:
            return xs
        gated = []
        for t in xs:
            mean_map = t.mean(dim=1, keepdim=True)
            max_map = t.amax(dim=1, keepdim=True)
            gate = torch.sigmoid(self.spatial_gate(torch.cat([mean_map, max_map], dim=1)))
            gated.append(t * (1.0 + self.eta * gate))
        return gated

    def forward(self, x: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        xs = self._align_size(self._as_list(x))
        self._check_channels(xs)
        xs = self._apply_spatial_gate(xs)

        static_w = F.relu(self.static_w, inplace=False)
        static_logits = torch.log(static_w + self.eps).view(1, self.n_inputs)

        if self.use_dynamic:
            pooled = torch.cat([F.adaptive_avg_pool2d(t, 1).flatten(1) for t in xs], dim=1)
            dynamic_logits = self.gate(pooled)
        else:
            dynamic_logits = torch.zeros(
                xs[0].shape[0],
                self.n_inputs,
                device=xs[0].device,
                dtype=xs[0].dtype,
            )

        alpha = torch.softmax(static_logits.to(dynamic_logits.dtype) + self.gamma * dynamic_logits, dim=1)
        fused = sum(alpha[:, i].view(-1, 1, 1, 1) * xs[i] for i in range(self.n_inputs))
        return self.out_conv(fused)

    def extra_repr(self) -> str:
        return (
            f"c1={self.c1}, c2={self.c2}, n_inputs={self.n_inputs}, "
            f"use_dynamic={self.use_dynamic}, use_spatial={self.use_spatial}, eps={self.eps}"
        )
