"""Shape-matched YOLO11n to SA-DWPN-B weight-transfer check.

This script reports how many pretrained tensors can be reused. It does not
train and does not force mismatched tensors into the custom model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def default_model_path() -> str:
    repo_yaml = Path("experiments/yolo11n_sa_dwpn_b.yaml")
    source_yaml = Path("ultralytics/cfg/models/11/yolo11n-sa-dwpn.yaml")
    return str(source_yaml if source_yaml.exists() else repo_yaml)


def load_checkpoint_state_dict(weights: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"].float().state_dict()
    if hasattr(ckpt, "float") and hasattr(ckpt.float(), "state_dict"):
        return ckpt.float().state_dict()
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check shape-matched transfer from YOLO11n to SA-DWPN-B.")
    parser.add_argument("--model", default=default_model_path(), help="Path to SA-DWPN YAML.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Path to pretrained YOLO11n weights.")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Pretrained weights not found: {args.weights}")

    model = YOLO(args.model)
    new_sd = model.model.state_dict()
    pre_sd = load_checkpoint_state_dict(args.weights)

    matched = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in pre_sd.items():
        if key not in new_sd:
            skipped_missing.append(key)
            continue
        if new_sd[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(new_sd[key].shape)))
            continue
        matched[key] = value

    result = model.model.load_state_dict(matched, strict=False)
    missing_after_load = list(result.missing_keys)
    unexpected_after_load = list(result.unexpected_keys)

    print(f"custom model params: {len(new_sd)} tensors")
    print(f"pretrained params: {len(pre_sd)} tensors")
    print(f"loaded keys: {len(matched)}")
    print(f"skipped keys because of shape mismatch: {len(skipped_shape)}")
    for key, old_shape, new_shape in skipped_shape[:80]:
        print(f"  shape mismatch: {key}: pretrained={old_shape}, custom={new_shape}")
    print(f"skipped keys missing in custom model: {len(skipped_missing)}")
    print(f"missing keys after load: {len(missing_after_load)}")
    for key in missing_after_load[:120]:
        print(f"  missing: {key}")
    print(f"unexpected keys after load: {len(unexpected_after_load)}")


if __name__ == "__main__":
    main()
