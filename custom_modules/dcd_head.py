"""Original Dual Context Decoupled detection head adapted to YOLO11.

The implementation preserves the DCD spatial-context construction while
retaining Ultralytics' native anchor-free prediction, DFL, losses, assignment,
decoding, and post-processing interfaces.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


class DCDDetect(Detect):
    """Dual-context spatially decoupled head for three YOLO11 detection levels.

    Input order is ``[P2_aux, P3, P4, P5]``. P2 is used only as the lower
    regression context for P3; predictions remain limited to P3/P4/P5.
    """

    def __init__(
        self,
        nc: int = 80,
        context_channels: int = 256,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
    ) -> None:
        if end2end:
            raise ValueError("DCDDetect supports the standard one-to-many YOLO11 Detect path only.")
        if len(ch) != 4:
            raise ValueError(
                "DCDDetect expects [P2 auxiliary, P3, P4, P5] channel inputs; "
                f"received {len(ch)} inputs."
            )
        if context_channels <= 0 or context_channels % 2:
            raise ValueError("context_channels must be a positive even integer.")

        # Detect owns the native DFL/inference interface and sees only the
        # three true prediction levels. Its temporary cv2/cv3 towers are
        # replaced below by the original DCD task-specific paths.
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=False,
            ch=tuple(ch[1:]),
        )
        self.context_channels = int(context_channels)

        self.input_projections = nn.ModuleList(
            Conv(input_channels, self.context_channels, k=1)
            for input_channels in ch
        )

        # YOLO11's P3-P5 pyramid has no deeper P6 neighbor. This stride-2
        # convolution creates a P6 context only for the P5 DCD equations; it
        # is neither an additional paper module nor a fourth detection level.
        self.p6_context_adapter = Conv(
            self.context_channels,
            self.context_channels,
            k=3,
            s=2,
        )

        # Classification context: Concat(DC(A_i), A_{i+1}). The two 3x3
        # convolutions preserve 2C channels, and PixelShuffle(2) implements
        # the original YOLOv5 channel-to-space Expand operation exactly.
        classification_channels = 2 * self.context_channels
        expanded_channels = classification_channels // 4
        self.cls_downsample = nn.ModuleList(
            Conv(self.context_channels, self.context_channels, k=3, s=2)
            for _ in range(self.nl)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(classification_channels, classification_channels, k=3),
                Conv(classification_channels, classification_channels, k=3),
                nn.PixelShuffle(2),
                nn.Conv2d(expanded_channels, self.nc, kernel_size=1),
            )
            for _ in range(self.nl)
        )

        # Regression context:
        # DC(A_{i-1} + U(A_i)) + A_i + U(A_{i+1}). Classification and
        # regression use separate downsampling/context towers.
        self.reg_lower_downsample = nn.ModuleList(
            Conv(self.context_channels, self.context_channels, k=3, s=2)
            for _ in range(self.nl)
        )
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(self.context_channels, self.context_channels, k=3),
                Conv(self.context_channels, self.context_channels, k=3),
                nn.Conv2d(self.context_channels, 4 * self.reg_max, kernel_size=1),
            )
            for _ in range(self.nl)
        )

    @staticmethod
    def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Nearest-align spatial shape only when odd-sized pyramids require it."""

        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(source, size=target.shape[-2:], mode="nearest")

    def _build_contexts(
        self,
        inputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Construct independent classification and regression contexts."""

        if len(inputs) != 4:
            raise ValueError(
                "DCDDetect forward expects [P2 auxiliary, P3, P4, P5]; "
                f"received {len(inputs)} tensors."
            )

        aligned = [projection(feature) for projection, feature in zip(self.input_projections, inputs)]
        p2, p3, p4, p5 = aligned
        p6_context = self.p6_context_adapter(p5)
        pyramid = [p2, p3, p4, p5, p6_context]
        detection_features = list(inputs[1:])

        classification_contexts: list[torch.Tensor] = []
        regression_contexts: list[torch.Tensor] = []
        for level in range(self.nl):
            lower = pyramid[level]
            current = pyramid[level + 1]
            upper = pyramid[level + 2]

            current_down = self.cls_downsample[level](current)
            upper_for_classification = self._resize_like(upper, current_down)
            classification_contexts.append(
                torch.cat((current_down, upper_for_classification), dim=1)
            )

            current_up = self._resize_like(current, lower)
            lower_detail = self.reg_lower_downsample[level](lower + current_up)
            lower_detail = self._resize_like(lower_detail, current)
            upper_semantics = self._resize_like(upper, current)
            regression_contexts.append(lower_detail + current + upper_semantics)

        return classification_contexts, regression_contexts, detection_features

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build DCD contexts and return the native YOLO11 prediction mapping."""

        if box_head is None or cls_head is None:
            return dict()

        classification_contexts, regression_contexts, detection_features = self._build_contexts(x)
        batch_size = detection_features[0].shape[0]
        box_outputs: list[torch.Tensor] = []
        class_outputs: list[torch.Tensor] = []
        for level in range(self.nl):
            box_prediction = box_head[level](regression_contexts[level])
            class_prediction = cls_head[level](classification_contexts[level])
            class_prediction = self._resize_like(class_prediction, detection_features[level])
            box_outputs.append(box_prediction.view(batch_size, 4 * self.reg_max, -1))
            class_outputs.append(class_prediction.view(batch_size, self.nc, -1))

        return {
            "boxes": torch.cat(box_outputs, dim=-1),
            "scores": torch.cat(class_outputs, dim=-1),
            "feats": detection_features,
        }


__all__ = ["DCDDetect"]
