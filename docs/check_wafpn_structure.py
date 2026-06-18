"""Reference structure check for WAFPN-v1 after Ultralytics integration.

Run this script only inside an environment where Ultralytics has already been
patched to import and parse AlignWeightedAdd2. This is not a training script.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect


ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "experiments" / "yolo11n_wafpn_v1_640.yaml"


def iter_modules(model):
    base = getattr(model, "model", model)
    module = getattr(base, "model", base)
    return module.modules()


def main() -> None:
    print(f"Loading model YAML: {YAML_PATH}")
    model = YOLO(str(YAML_PATH))
    model.info()

    modules = list(iter_modules(model))
    detect_layers = [m for m in modules if isinstance(m, Detect)]
    if len(detect_layers) != 1:
        raise RuntimeError(f"Expected one Detect module, found {len(detect_layers)}")

    detect = detect_layers[0]
    nl = getattr(detect, "nl", None)
    if nl != 3:
        raise RuntimeError(f"Expected 3 Detect heads (P3/P4/P5), got {nl}")
    print("Detect heads: 3 (P3/P4/P5)")

    wafpn_modules = [m for m in modules if m.__class__.__name__ == "AlignWeightedAdd2"]
    print(f"AlignWeightedAdd2 modules: {len(wafpn_modules)}")
    if len(wafpn_modules) != 4:
        raise RuntimeError(f"Expected 4 AlignWeightedAdd2 modules, found {len(wafpn_modules)}")

    weighted_add_modules = [m for m in modules if m.__class__.__name__ == "WeightedAdd2"]
    for idx, module in enumerate(weighted_add_modules):
        weights = getattr(module, "w", None)
        if weights is not None:
            print(f"WeightedAdd2[{idx}] weights: {weights.detach().cpu().tolist()}")

    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        _ = model.model(dummy)
    print("Dummy forward: success")


if __name__ == "__main__":
    main()
