"""Visualize original, ERUP, VGUP, and the VGUP spatial gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.erup import ERUPPreprocessor
from custom_modules.vgup import VGUPPreprocessor


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/erup_vgup/visualization.png",
    )
    args = parser.parse_args()
    image = load_image(args.image)
    erup = ERUPPreprocessor().eval()
    vgup = VGUPPreprocessor().eval()
    with torch.inference_mode():
        erup_output, _ = erup(image, return_debug=True)
        vgup_output, vgup_debug = vgup(image, return_debug=True)
    gate = vgup_debug["spatial_gate"].squeeze().cpu().numpy()
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(to_image(image))
    axes[0].set_title("Original")
    axes[1].imshow(to_image(erup_output))
    axes[1].set_title("ERUP")
    axes[2].imshow(to_image(vgup_output))
    axes[2].set_title("VGUP")
    heatmap = axes[3].imshow(gate, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("VGUP spatial gate")
    figure.colorbar(heatmap, ax=axes[3], fraction=0.046)
    for axis in axes:
        axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
