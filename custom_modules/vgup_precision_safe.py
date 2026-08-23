"""Numerically stable full VGUP adapter for mixed-precision host models.

The module preserves the complete VGUP graph (BPW, KBL, global gate and
spatial gate). Only the numerically sensitive image-filtering operations are
evaluated in FP32; the encoder and the downstream detector remain controlled
by the surrounding AMP context.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from custom_modules.erup import _tensor_stats
from custom_modules.vgup import VGUPPreprocessor


class VGUPPrecisionSafePreprocessor(VGUPPreprocessor):
    """Full VGUP with an FP32 precision island around BPW/KBL filtering."""

    def forward(
        self,
        image: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        predictions = self.encoder(image)
        input_dtype = image.dtype

        if image.device.type in {"cpu", "cuda"}:
            precision_context = torch.autocast(
                device_type=image.device.type,
                enabled=False,
            )
        else:
            precision_context = nullcontext()

        with precision_context:
            image_fp32 = image.float()
            bpw_params_fp32 = predictions["bpw_params"].float()
            kbl_params_fp32 = predictions["kbl_params"].float()
            global_gate_fp32 = predictions["global_gate"].float()
            spatial_gate_fp32 = predictions["spatial_gate"].float()

            enhanced_bpw = self.bpw(image_fp32, bpw_params_fp32)
            gated_bpw = self.apply_bpw_gate(
                image_fp32,
                enhanced_bpw,
                global_gate_fp32,
            )
            enhanced_kbl = self.kbl(gated_bpw, kbl_params_fp32)
            output_fp32 = self.apply_kbl_gate(
                gated_bpw,
                enhanced_kbl,
                spatial_gate_fp32,
            )

        output = output_fp32.to(dtype=input_dtype)
        if not return_debug:
            return output

        debug = {
            **predictions,
            "bpw_param_stats": _tensor_stats(predictions["bpw_params"]),
            "kbl_param_stats": _tensor_stats(predictions["kbl_params"]),
            "global_gate_stats": _tensor_stats(predictions["global_gate"]),
            "spatial_gate_stats": _tensor_stats(predictions["spatial_gate"]),
            "bpw_image": enhanced_bpw,
            "gated_bpw_image": gated_bpw,
            "kbl_image": enhanced_kbl,
            "output_image": output,
        }
        return output, debug


__all__ = ["VGUPPrecisionSafePreprocessor"]
