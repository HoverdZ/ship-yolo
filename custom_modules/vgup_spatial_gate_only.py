"""Controlled VGUP ablation that removes only the global gate.

The original :class:`VGUPPreprocessor` and the existing global-gate-only
variant remain unchanged. This independent module retains BPW, KBL, the
shared lightweight encoder, and the spatial KBL residual gate. The global
BPW gate head is not instantiated, so BPW is accepted without global gating.
"""

from __future__ import annotations

from custom_modules.vgup import VGUPPreprocessor


class VGUPSpatialGateOnlyPreprocessor(VGUPPreprocessor):
    """VGUP with BPW/KBL and the spatial gate, but without a global gate."""

    def __init__(
        self,
        in_channels: int = 3,
        bpw_segments: int = 8,
        prediction_size: int = 128,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            bpw_segments=bpw_segments,
            prediction_size=prediction_size,
            use_global_gate=False,
            use_spatial_gate=True,
        )


__all__ = ["VGUPSpatialGateOnlyPreprocessor"]
