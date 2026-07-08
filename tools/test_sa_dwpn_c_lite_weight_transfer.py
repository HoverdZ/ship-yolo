"""Weight-transfer checks for YOLO11n-SA-DWPN-C-lite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_sa_dwpn_modules


def default_model_path() -> str:
    source_yaml = Path("ultralytics/cfg/models/11/yolo11n-sa-dwpn-c-lite.yaml")
    repo_yaml = Path("experiments/yolo11n_sa_dwpn_c_lite.yaml")
    return str(source_yaml if source_yaml.exists() else repo_yaml)


def load_state_dict(weights: str) -> dict[str, torch.Tensor]:
    import torch

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"].float().state_dict()
    if hasattr(ckpt, "float") and hasattr(ckpt.float(), "state_dict"):
        return ckpt.float().state_dict()
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check weight transfer into YOLO11n-SA-DWPN-C-lite.")
    parser.add_argument("--model", default=default_model_path(), help="Path to C-lite YAML.")
    parser.add_argument("--weights", required=True, help="Source weights, e.g. yolo11n.pt or SA-DWPN-B best.pt.")
    parser.add_argument("--min-matched", type=int, default=1, help="Fail if matched tensor count is below this value.")
    args = parser.parse_args()

    import torch

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Source weights not found: {args.weights}")

    register_sa_dwpn_modules()
    from ultralytics import YOLO

    target = YOLO(args.model)
    target_sd = target.model.state_dict()
    source_sd = load_state_dict(args.weights)

    matched = {}
    shape_mismatch = []
    missing_in_target = []
    for key, value in source_sd.items():
        if key not in target_sd:
            missing_in_target.append(key)
            continue
        if target_sd[key].shape != value.shape:
            shape_mismatch.append((key, tuple(value.shape), tuple(target_sd[key].shape)))
            continue
        matched[key] = value

    result = target.model.load_state_dict(matched, strict=False)
    missing_after_load = list(result.missing_keys)
    unexpected_after_load = list(result.unexpected_keys)
    spatial_only_new = [
        key
        for key in missing_after_load
        if "spatial_gate" in key or key.endswith(".eta")
    ]

    print(f"source keys: {len(source_sd)}")
    print(f"target keys: {len(target_sd)}")
    print(f"matched keys: {len(matched)}")
    print(f"shape mismatch: {len(shape_mismatch)}")
    for key, source_shape, target_shape in shape_mismatch[:80]:
        print(f"  shape mismatch: {key}: source={source_shape}, target={target_shape}")
    print(f"missing keys: {len(missing_after_load)}")
    for key in missing_after_load[:120]:
        print(f"  missing: {key}")
    print(f"unexpected keys: {len(unexpected_after_load)}")
    for key in unexpected_after_load[:80]:
        print(f"  unexpected: {key}")
    print(f"source keys missing in target: {len(missing_in_target)}")
    print(f"spatial-only new keys: {len(spatial_only_new)}")
    for key in spatial_only_new:
        print(f"  spatial-only new: {key}")

    if len(matched) < args.min_matched:
        raise AssertionError(f"Matched key count {len(matched)} is below --min-matched {args.min_matched}")


if __name__ == "__main__":
    main()
