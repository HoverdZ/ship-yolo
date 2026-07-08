"""YOLO detection targets for Grad-CAM style analysis."""

from __future__ import annotations


class ShipDetectionScoreTarget:
    """Target based on top ship detection scores rather than classification logits."""

    def __init__(self, class_id: int = 0, topk: int = 10, min_conf: float = 0.25) -> None:
        self.class_id = class_id
        self.topk = topk
        self.min_conf = min_conf

    def __call__(self, model_output):
        import torch

        tensors = model_output if isinstance(model_output, (list, tuple)) else [model_output]
        scores = []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            flat = tensor.flatten()
            if flat.numel():
                scores.append(flat.topk(min(self.topk, flat.numel())).values.mean())
        if not scores:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(scores).mean()
