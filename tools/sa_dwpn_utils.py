"""Shared helpers for SA-DWPN experiment scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "sa_dwpn_protocol.yaml"

VARIANTS = {
    "b": {
        "name": "yolo11n_sa_dwpn_b_640",
        "yaml": ROOT / "experiments" / "yolo11n_sa_dwpn_b.yaml",
        "spatial_positions": [],
    },
    "c_lite": {
        "name": "yolo11n_sa_dwpn_c_lite_640",
        "yaml": ROOT / "experiments" / "yolo11n_sa_dwpn_c_lite.yaml",
        "spatial_positions": [2, 3],
    },
    "t3_only": {
        "name": "yolo11n_sa_dwpn_c_t3_only_640",
        "yaml": ROOT / "experiments" / "yolo11n_sa_dwpn_c_t3_only.yaml",
        "spatial_positions": [2],
    },
    "o3_only": {
        "name": "yolo11n_sa_dwpn_c_o3_only_640",
        "yaml": ROOT / "experiments" / "yolo11n_sa_dwpn_c_o3_only.yaml",
        "spatial_positions": [3],
    },
}


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = read_yaml(path)
    required = ["training", "initialization", "constraints", "artifact_policy"]
    missing = [key for key in required if key not in protocol]
    if missing:
        raise KeyError(f"Protocol missing required sections: {missing}")
    return protocol


def protocol_hash(path: str | Path = PROTOCOL_PATH) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    return VARIANTS[variant]


def all_model_yaml_paths() -> dict[str, Path]:
    return {name: cfg["yaml"] for name, cfg in VARIANTS.items()}


def combined_layers(model_yaml: dict[str, Any]) -> list[list[Any]]:
    return list(model_yaml["backbone"]) + list(model_yaml["head"])


def sdwf_layers(model_yaml: dict[str, Any]) -> list[tuple[int, list[Any]]]:
    return [(idx, layer) for idx, layer in enumerate(combined_layers(model_yaml)) if layer[2] == "SDWF"]


def spatial_positions(model_yaml: dict[str, Any]) -> list[int]:
    positions: list[int] = []
    for order, (_idx, layer) in enumerate(sdwf_layers(model_yaml), start=1):
        args = layer[3]
        if len(args) >= 4 and bool(args[3]):
            positions.append(order)
    return positions


def detect_from(model_yaml: dict[str, Any]) -> list[int]:
    detect_layers = [layer for layer in combined_layers(model_yaml) if layer[2] == "Detect"]
    if len(detect_layers) != 1:
        raise AssertionError(f"Expected one Detect layer, found {len(detect_layers)}")
    return list(detect_layers[0][0])


def validate_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Invalid checkpoint type for resume: {type(ckpt).__name__}")
    required = ["epoch", "optimizer", "train_args"]
    missing = [key for key in required if key not in ckpt or ckpt[key] is None]
    if missing:
        raise RuntimeError(f"Invalid resume checkpoint {path}; missing {missing}")
    return {"path": str(path), "epoch": ckpt.get("epoch"), "has_optimizer": ckpt.get("optimizer") is not None}


def shape_matched_transfer_report(model, weights: str | Path) -> dict[str, Any]:
    import torch

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        source_sd = ckpt["model"].float().state_dict()
    elif hasattr(ckpt, "float") and hasattr(ckpt.float(), "state_dict"):
        source_sd = ckpt.float().state_dict()
    elif isinstance(ckpt, dict):
        source_sd = ckpt
    else:
        raise TypeError(f"Unsupported weights format: {type(ckpt).__name__}")
    target_sd = model.model.state_dict()
    matched = [
        key for key, value in source_sd.items() if key in target_sd and tuple(target_sd[key].shape) == tuple(value.shape)
    ]
    mismatched = [
        key for key, value in source_sd.items() if key in target_sd and tuple(target_sd[key].shape) != tuple(value.shape)
    ]
    missing = [key for key in target_sd if key not in source_sd]
    return {
        "source_keys": len(source_sd),
        "target_keys": len(target_sd),
        "matched_keys": len(matched),
        "shape_mismatch": len(mismatched),
        "missing_keys": len(missing),
        "matched_examples": matched[:20],
        "shape_mismatch_examples": mismatched[:20],
        "missing_examples": missing[:20],
    }


def ensure_data_yaml(path: str | Path) -> Path:
    path = Path(path)
    if not str(path):
        raise ValueError("Data YAML must not be empty; refusing to fall back to coco8.yaml.")
    if not path.exists():
        raise FileNotFoundError(f"Data YAML not found: {path}; refusing to fall back to coco8.yaml.")
    return path
