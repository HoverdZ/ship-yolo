"""Export real Detect-input feature-energy maps from one trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _normalized_energy(feature: torch.Tensor) -> np.ndarray:
    value = feature[0].float().square().mean(dim=0).cpu().numpy()
    value -= value.min()
    value /= max(float(value.max()), 1e-12)
    return value


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
    detect = model.model.model[-1]
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.clear()
        captured.extend(value.detach().cpu() for value in inputs[0])

    handle = detect.register_forward_pre_hook(capture)
    try:
        model.predict(source=args.image, imgsz=args.imgsz, verbose=False)
    finally:
        handle.remove()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    original = Image.open(args.image).convert("RGB")
    original.save(output / "input_image.png")
    metadata = []
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        len(captured) + 1,
        figsize=(4 * (len(captured) + 1), 4),
        constrained_layout=True,
    )
    axes[0].imshow(original)
    axes[0].set_title("Input")
    axes[0].axis("off")
    for index, (feature, stride) in enumerate(
        zip(captured, detect.stride, strict=True),
        start=1,
    ):
        level = f"P{int(round(torch.log2(stride).item()))}"
        energy = _normalized_energy(feature)
        Image.fromarray(np.uint8(energy * 255)).save(
            output / f"{level}_feature_energy.png"
        )
        axes[index].imshow(energy, cmap="magma", vmin=0, vmax=1)
        axes[index].set_title(f"{level} (stride {int(stride)})")
        axes[index].axis("off")
        metadata.append(
            {
                "level": level,
                "stride": float(stride),
                "shape": list(feature.shape),
                "mean_abs": float(feature.abs().mean()),
                "energy": float(feature.square().mean()),
            }
        )
    figure.savefig(output / "pyramid_feature_panel.png", dpi=300)
    plt.close(figure)
    (output / "pyramid_feature_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
