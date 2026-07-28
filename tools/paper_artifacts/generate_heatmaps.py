"""CAM/feature-energy heatmaps for Ultralytics detection models.

Grad-CAM, Grad-CAM++, and EigenCAM are attempted honestly. Any incompatibility
is recorded and falls back to the deterministic feature-energy map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _tensor_from_image(path: Path, imgsz: int, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    original = Image.open(path).convert("RGB")
    resized = original.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, original


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for child in value.values():
            try:
                return _first_tensor(child)
            except LookupError:
                pass
    if isinstance(value, (list, tuple)):
        for child in value:
            try:
                return _first_tensor(child)
            except LookupError:
                pass
    raise LookupError("No tensor output was available for CAM.")


def _normalize(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    value -= value.amin(dim=(-2, -1), keepdim=True)
    return value / value.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def _energy(activation: torch.Tensor) -> torch.Tensor:
    return _normalize(activation.detach().float().square().mean(dim=1, keepdim=True))


def _eigen(activation: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = activation.shape
    maps = []
    for index in range(batch):
        matrix = activation[index].detach().float().reshape(channels, -1).T
        matrix -= matrix.mean(dim=0, keepdim=True)
        _, _, vectors = torch.linalg.svd(matrix, full_matrices=False)
        maps.append((matrix @ vectors[0]).reshape(1, height, width).abs())
    return _normalize(torch.stack(maps))


def _grad_cam(activation: torch.Tensor, gradient: torch.Tensor, plus_plus: bool) -> torch.Tensor:
    if plus_plus:
        gradient2 = gradient.square()
        gradient3 = gradient2 * gradient
        denominator = 2 * gradient2 + (activation * gradient3).sum(dim=(-2, -1), keepdim=True)
        alpha = gradient2 / denominator.clamp_min(1e-8)
        weights = (alpha * F.relu(gradient)).sum(dim=(-2, -1), keepdim=True)
    else:
        weights = gradient.mean(dim=(-2, -1), keepdim=True)
    return _normalize(F.relu((weights * activation).sum(dim=1, keepdim=True)))


def _overlay(original: Image.Image, heatmap: torch.Tensor) -> Image.Image:
    heat = heatmap[0, 0].detach().cpu().numpy()
    heat = np.asarray(Image.fromarray(np.uint8(heat * 255)).resize(original.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255
    base = np.asarray(original, dtype=np.float32) / 255
    color = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=-1)
    return Image.fromarray(np.uint8(np.clip(0.55 * base + 0.45 * color, 0, 1) * 255))


def generate_one(model, image_path: str | Path, target_layer: int, output: str | Path, method: str = "gradcam", imgsz: int = 640) -> dict[str, Any]:
    network = model.model.eval()
    device = next(network.parameters()).device
    tensor, original = _tensor_from_image(Path(image_path), imgsz, device)
    layer = network.model[target_layer]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, result) -> None:
        activation = _first_tensor(result)
        captured["activation"] = activation
        if activation.requires_grad:
            activation.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = layer.register_forward_hook(hook)
    requested = method.lower()
    used = requested
    failure = None
    try:
        tensor.requires_grad_(requested in {"gradcam", "gradcam++"})
        output_value = network(tensor)
        activation = captured["activation"]
        if requested in {"gradcam", "gradcam++"}:
            prediction = _first_tensor(output_value)
            target = prediction.float().amax()
            network.zero_grad(set_to_none=True)
            target.backward()
            heatmap = _grad_cam(activation, captured["gradient"], requested == "gradcam++")
        elif requested == "eigencam":
            heatmap = _eigen(activation)
        elif requested == "feature-energy":
            heatmap = _energy(activation)
        else:
            raise ValueError(f"Unknown heatmap method: {method}")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        used = "feature-energy"
        if "activation" not in captured:
            raise
        heatmap = _energy(captured["activation"])
    finally:
        handle.remove()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _overlay(original, heatmap).save(destination)
    report = {
        "image": str(image_path),
        "target_layer": target_layer,
        "target_module": type(layer).__name__,
        "requested_method": requested,
        "used_method": used,
        "fallback_reason": failure,
        "output": str(destination),
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["gradcam", "gradcam++", "eigencam", "feature-energy"], default="gradcam")
    args = parser.parse_args()
    from ultralytics import YOLO

    print(generate_one(YOLO(args.weights), args.image, args.layer, args.output, args.method))


if __name__ == "__main__":
    main()

