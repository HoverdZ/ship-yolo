"""Export CA-SCAM local-contrast and calibrated attention heatmaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.calibrated_scam_utils import CA_LAYER_INDICES, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="artifacts/ca_scam_visualization")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor, resize

    model = build_model()
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
        state = checkpoint["model"].float().state_dict()
        model.model.load_state_dict(state, strict=True)
    network = model.model.cpu().eval()
    image = Image.open(args.image).convert("RGB")
    tensor = pil_to_tensor(image).float().div(255).unsqueeze(0)
    tensor = resize(tensor, [args.imgsz, args.imgsz], antialias=True)
    captured: dict[int, torch.Tensor] = {}
    hooks = [
        network.model[index].register_forward_pre_hook(
            lambda _module, values, index=index: captured.__setitem__(
                index,
                values[0].detach(),
            )
        )
        for index in CA_LAYER_INDICES
    ]
    try:
        with torch.inference_mode():
            network(tensor)
    finally:
        for hook in hooks:
            hook.remove()

    import matplotlib.pyplot as plt

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for index in CA_LAYER_INDICES:
        module = network.model[index]
        local, contrast_map, beta = module.contrast_state(captured[index])
        metadata[str(index)] = {
            "feature_shape": list(captured[index].shape),
            "beta": float(beta.item()),
            "local_contrast_min": float(local.min()),
            "local_contrast_max": float(local.max()),
            "contrast_map_min": float(contrast_map.min()),
            "contrast_map_max": float(contrast_map.max()),
        }
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(local[0, 0].cpu(), cmap="magma")
        axes[0].set_title(f"Layer {index}: local contrast")
        axes[1].imshow(contrast_map[0, 0].cpu(), cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title(f"Layer {index}: CA map")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / f"ca_scam_layer_{index}.png", dpi=180)
        plt.close(figure)
    (output_dir / "ca_scam_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
