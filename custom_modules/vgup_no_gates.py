"""Controlled VGUP ablation that removes both residual gates.

The original VGUP and both one-gate variants remain unchanged. This module
retains the shared lightweight encoder and the complete BPW -> KBL processing
path, but instantiates neither the global gate head nor the spatial gate head.
Both processed images are therefore accepted directly.
"""

from __future__ import annotations

from custom_modules.vgup import VGUPPreprocessor


class VGUPNoGatesPreprocessor(VGUPPreprocessor):
    """VGUP with BPW/KBL retained and both residual gates removed."""

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
            use_spatial_gate=False,
        )


__all__ = ["VGUPNoGatesPreprocessor"]
