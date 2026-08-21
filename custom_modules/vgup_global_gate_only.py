"""Controlled VGUP ablation that removes only the spatial gate.

The original :class:`VGUPPreprocessor` remains unchanged. This independent
module retains the shared lightweight encoder, BPW and KBL operators, and the
global residual acceptance gate. The spatial gate head is not instantiated;
consequently KBL is applied directly after globally gated BPW processing.
"""

from __future__ import annotations

from custom_modules.vgup import VGUPPreprocessor


class VGUPGlobalGateOnlyPreprocessor(VGUPPreprocessor):
    """VGUP with BPW/KBL and the global gate, but without a spatial gate."""

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
            use_global_gate=True,
            use_spatial_gate=False,
        )


__all__ = ["VGUPGlobalGateOnlyPreprocessor"]
