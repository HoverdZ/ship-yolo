"""Export the full trained VGUP processing path from a real forward hook."""

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


def _image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor[0]
        .detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.uint8(array * 255))


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
    preprocessor = model.model.model[0]
    if type(preprocessor).__name__ != "VGUPPreprocessor":
        raise ValueError("Checkpoint does not start with VGUPPreprocessor.")
    captured: list[torch.Tensor] = []
    handle = preprocessor.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        prediction = model.predict(
            source=args.image,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
    finally:
        handle.remove()
    with torch.inference_mode():
        _output, debug = preprocessor(captured[-1], return_debug=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, tensor in {
        "input": captured[-1],
        "bpw_raw": debug["bpw_image"],
        "bpw_global_gated": debug["gated_bpw_image"],
        "kbl_raw": debug["kbl_image"],
        "vgup_final": debug["output_image"],
    }.items():
        _image(tensor).save(output / f"{name}.png")
    gate = debug["spatial_gate"][0, 0].detach().cpu().numpy()
    Image.fromarray(np.uint8(gate * 255)).save(
        output / "spatial_visibility_gate.png"
    )
    prediction.save(filename=str(output / "final_model_prediction.jpg"))
    stats = {
        "global_gate": float(debug["global_gate"].mean()),
        "spatial_gate_mean": float(debug["spatial_gate"].mean()),
        "spatial_gate_std": float(debug["spatial_gate"].std()),
        "spatial_gate_min": float(debug["spatial_gate"].min()),
        "spatial_gate_max": float(debug["spatial_gate"].max()),
        "bpw_parameter_mean": float(debug["bpw_params"].mean()),
        "kbl_parameter_mean": float(debug["kbl_params"].mean()),
    }
    (output / "vgup_forward_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
