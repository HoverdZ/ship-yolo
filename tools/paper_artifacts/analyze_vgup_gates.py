"""Measure trained VGUP gates against objective image visibility statistics."""

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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(args.weights)
    preprocessor = model.model.model[0]
    if type(preprocessor).__name__ != "VGUPPreprocessor":
        raise ValueError("Checkpoint does not start with VGUPPreprocessor.")
    paths = sorted(
        item
        for item in Path(args.images).rglob("*")
        if item.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit > 0:
        paths = paths[: args.limit]
    rows = []
    for path in paths:
        captured: list[torch.Tensor] = []
        handle = preprocessor.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            model.predict(source=str(path), imgsz=args.imgsz, verbose=False)
        finally:
            handle.remove()
        with torch.inference_mode():
            _output, debug = preprocessor(captured[-1], return_debug=True)
        gray = captured[-1].mean(dim=1, keepdim=True)
        local_mean = torch.nn.functional.avg_pool2d(
            gray,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        rows.append(
            {
                "image": path.name,
                "global_gate": float(debug["global_gate"].mean()),
                "spatial_gate_mean": float(debug["spatial_gate"].mean()),
                "spatial_gate_std": float(debug["spatial_gate"].std()),
                "brightness_mean": float(gray.mean()),
                "global_contrast_std": float(gray.std()),
                "local_contrast_mean": float((gray - local_mean).abs().mean()),
            }
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "vgup_gate_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "vgup_gate_statistics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if rows:
        import matplotlib.pyplot as plt

        arrays = {key: np.array([row[key] for row in rows]) for key in rows[0] if key != "image"}
        figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axes[0, 0].hist(arrays["global_gate"], bins=30)
        axes[0, 0].set_title("Global gate distribution")
        axes[0, 1].boxplot(
            [arrays["global_gate"], arrays["spatial_gate_mean"]],
            labels=["global", "spatial mean"],
        )
        axes[0, 1].set_title("Gate values")
        axes[1, 0].scatter(
            arrays["brightness_mean"],
            arrays["global_gate"],
            s=12,
        )
        axes[1, 0].set(xlabel="Brightness", ylabel="Global gate")
        axes[1, 1].scatter(
            arrays["local_contrast_mean"],
            arrays["spatial_gate_mean"],
            s=12,
        )
        axes[1, 1].set(
            xlabel="Local contrast",
            ylabel="Spatial gate mean",
        )
        figure.savefig(output / "vgup_gate_distributions.png", dpi=300)
        plt.close(figure)
    print(csv_path)


if __name__ == "__main__":
    main()
