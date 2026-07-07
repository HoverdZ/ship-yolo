"""Build checks for YOLO11n-SA-DWPN-C-lite.

This script validates both C-lite and the unchanged B scaffold. It does not
train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_sa_dwpn_modules


EXPECTED_SPATIAL = {21: "T3", 22: "O3"}
EXPECTED_SDWF = {18: "T4", 21: "T3", 22: "O3", 24: "O4", 26: "O5"}


def default_c_model_path() -> str:
    source_yaml = Path("ultralytics/cfg/models/11/yolo11n-sa-dwpn-c-lite.yaml")
    repo_yaml = Path("experiments/yolo11n_sa_dwpn_c_lite.yaml")
    return str(source_yaml if source_yaml.exists() else repo_yaml)


def default_b_model_path() -> str:
    source_yaml = Path("ultralytics/cfg/models/11/yolo11n-sa-dwpn.yaml")
    repo_yaml = Path("experiments/yolo11n_sa_dwpn_b.yaml")
    return str(source_yaml if source_yaml.exists() else repo_yaml)


def layer_index(module) -> int | None:
    value = getattr(module, "i", None)
    return int(value) if isinstance(value, int) else None


def validate_model(model, *, expected_spatial: set[int], label: str) -> tuple[int, float | None]:
    model.info(verbose=True)
    layers = list(model.model.modules())
    sdwf = [m for m in layers if m.__class__.__name__ == "SDWF"]
    dwdown = [m for m in layers if m.__class__.__name__ == "DWDown"]
    detects = [m for m in layers if m.__class__.__name__ == "Detect"]

    if len(sdwf) != 5:
        raise AssertionError(f"{label}: expected 5 SDWF modules, found {len(sdwf)}")
    if len(dwdown) != 1:
        raise AssertionError(f"{label}: expected 1 DWDown module, found {len(dwdown)}")
    if len(detects) != 1:
        raise AssertionError(f"{label}: expected 1 Detect module, found {len(detects)}")

    detect = detects[0]
    if getattr(detect, "nl", None) != 3:
        raise AssertionError(f"{label}: expected 3 Detect inputs, got {getattr(detect, 'nl', None)}")

    sdwf_by_idx = {layer_index(m): m for m in sdwf}
    missing = set(EXPECTED_SDWF) - set(sdwf_by_idx)
    if missing:
        raise AssertionError(f"{label}: missing SDWF layer indices: {sorted(missing)}")

    enabled = {idx for idx, module in sdwf_by_idx.items() if getattr(module, "use_spatial", False)}
    if enabled != expected_spatial:
        raise AssertionError(f"{label}: expected spatial gates {sorted(expected_spatial)}, got {sorted(enabled)}")

    if expected_spatial and enabled != {21, 22}:
        raise AssertionError(f"{label}: spatial gates must be enabled only at T3 and O3")

    detect_from = getattr(detect, "f", None)
    if list(detect_from) != [22, 24, 26]:
        raise AssertionError(f"{label}: expected Detect from [22, 24, 26], got {detect_from}")

    params = sum(p.numel() for p in model.model.parameters())
    gflops = getattr(model.model, "gflops", None)
    print(f"{label}: build OK")
    print(f"{label}: params={params}")
    print(f"{label}: GFLOPs={gflops}")
    print(f"{label}: spatial gates={[(idx, EXPECTED_SDWF[idx]) for idx in sorted(enabled)]}")
    return params, gflops


def main() -> None:
    parser = argparse.ArgumentParser(description="Build-check YOLO11n-SA-DWPN-C-lite.")
    parser.add_argument("--model", default=default_c_model_path(), help="Path to C-lite YAML.")
    parser.add_argument("--b-model", default=default_b_model_path(), help="Path to B YAML.")
    args = parser.parse_args()

    register_sa_dwpn_modules()
    from ultralytics import YOLO

    c_model = YOLO(args.model)
    c_params, c_gflops = validate_model(c_model, expected_spatial={21, 22}, label="C-lite")

    b_model = YOLO(args.b_model)
    b_params, b_gflops = validate_model(b_model, expected_spatial=set(), label="B")

    print(f"param delta C-lite - B: {c_params - b_params}")
    if c_gflops is not None and b_gflops is not None:
        print(f"GFLOPs delta C-lite - B: {c_gflops - b_gflops}")


if __name__ == "__main__":
    main()
