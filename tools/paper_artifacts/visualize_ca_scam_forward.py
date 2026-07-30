"""Export CA-SCAM evidence from real forward hooks, never synthetic heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _save_map(tensor: torch.Tensor, path: Path) -> None:
    value = tensor.detach().float()
    if value.ndim == 4 and value.shape[1] != 1:
        value = value.square().mean(dim=1, keepdim=True)
    array = value[0, 0].cpu().numpy()
    array -= array.min()
    array /= max(float(array.max()), 1e-12)
    Image.fromarray(np.uint8(array * 255)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(args.weights)
    network = model.model.eval()
    modules = [
        (index, layer)
        for index, layer in enumerate(network.model)
        if type(layer).__name__ in {
            "CASCAM",
            "CASCAMFixedBeta",
            "CASCAMUnbounded",
        }
    ]
    if len(modules) != 3:
        raise ValueError(f"Expected three CA-SCAM modules, found {len(modules)}")
    captured: dict[int, torch.Tensor] = {}
    handles = [
        layer.register_forward_pre_hook(
            lambda _module, inputs, index=index: captured.__setitem__(
                index,
                inputs[0].detach().clone(),
            )
        )
        for index, layer in modules
    ]
    try:
        model.predict(source=args.image, imgsz=args.imgsz, verbose=False)
    finally:
        for handle in handles:
            handle.remove()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    Image.open(args.image).convert("RGB").save(output / "input_image.png")
    rows = []
    for level, (index, layer) in enumerate(modules, start=2):
        feature = captured[index]
        with torch.inference_mode():
            delta = layer.compute_context_residual(feature)
            local, contrast = layer.contrast_map(feature)
            beta = layer.calibration_beta()
            calibrated = (1.0 + beta * contrast) * delta
            final = feature + calibrated
        prefix = output / f"P{level}"
        _save_map(feature, prefix.with_name(prefix.name + "_input_feature.png"))
        _save_map(delta, prefix.with_name(prefix.name + "_scam_residual.png"))
        _save_map(local, prefix.with_name(prefix.name + "_local_contrast.png"))
        _save_map(contrast, prefix.with_name(prefix.name + "_spatial_calibration.png"))
        _save_map(calibrated, prefix.with_name(prefix.name + "_calibrated_residual.png"))
        _save_map(final, prefix.with_name(prefix.name + "_final_feature.png"))
        rows.append(
            {
                "level": f"P{level}",
                "layer": index,
                "beta": float(beta),
                "local_contrast_mean": float(local.mean()),
                "local_contrast_std": float(local.std()),
                "calibration_mean": float(contrast.mean()),
                "residual_mean_abs": float(delta.abs().mean()),
                "calibrated_residual_mean_abs": float(calibrated.abs().mean()),
            }
        )
    with (output / "ca_scam_forward_statistics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "ca_scam_forward_statistics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
