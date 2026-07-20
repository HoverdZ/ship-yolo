"""Inspect the SCConv experiment topology and emit explicit PASS/FAIL checks."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_scconv_modules
from tools.scconv_utils import MODEL_YAML


EXPECTED_SC_LAYERS = [2, 4, 6, 8]
EXPECTED_TOP_LEVEL_STRIDE2 = [0, 1, 3, 5, 7, 17, 20]
EXPECTED_DETECT_INPUTS = [16, 19, 22]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_YAML))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _combined_yaml_layers(config: dict[str, Any]) -> list[list[Any]]:
    return list(config["backbone"]) + list(config["head"])


def _flatten_tensors(value: Any) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for item in value.values()
            for tensor in _flatten_tensors(item)
        ]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _flatten_tensors(item)]
    return []


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    backbone_length = len(config["backbone"])
    static_layers = _combined_yaml_layers(config)

    register_scconv_modules()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules.block import C3k2
    from ultralytics.nn.modules.conv import Conv

    from custom_modules.c3k2_scconv import C3k2_SCConv
    from custom_modules.scconv import ScConv

    wrapper = YOLO(str(model_path))
    baseline_path = (
        Path(ultralytics.__file__).resolve().parent
        / "cfg"
        / "models"
        / "11"
        / "yolo11.yaml"
    )
    baseline_config = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    expected_config = copy.deepcopy(baseline_config)
    for layer_index in EXPECTED_SC_LAYERS:
        expected_config["backbone"][layer_index][2] = "C3k2_SCConv"
    top_level = list(wrapper.model.model)
    sc_nodes = [
        {
            "layer": int(module.i),
            "region": "backbone" if int(module.i) < backbone_length else "neck",
            "scconv_count": sum(
                1 for child in module.modules() if isinstance(child, ScConv)
            ),
        }
        for module in top_level
        if isinstance(module, C3k2_SCConv)
    ]
    stride2_layers = [
        int(module.i)
        for module in top_level
        if isinstance(module, Conv) and tuple(module.conv.stride) == (2, 2)
    ]
    detect_modules = [
        module for module in top_level if module.__class__.__name__ == "Detect"
    ]
    if len(detect_modules) != 1:
        raise AssertionError(f"Expected exactly one Detect module, got {len(detect_modules)}.")
    detect = detect_modules[0]
    detect_feature_shapes: list[list[int]] = []

    def capture_detect_inputs(_module: Any, inputs: tuple[Any, ...]) -> None:
        detect_feature_shapes[:] = [list(tensor.shape) for tensor in inputs[0]]

    hook = detect.register_forward_pre_hook(capture_detect_inputs)
    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
    wrapper.model.to(args.device).eval()
    with torch.no_grad():
        output = wrapper.model(sample)
    hook.remove()
    output_tensors = _flatten_tensors(output)
    forward_finite = bool(output_tensors) and all(
        bool(torch.isfinite(tensor).all()) for tensor in output_tensors
    )
    expected_feature_shapes = [
        [1, 64, args.imgsz // 8, args.imgsz // 8],
        [1, 128, args.imgsz // 16, args.imgsz // 16],
        [1, 256, args.imgsz // 32, args.imgsz // 32],
    ]

    yaml_backbone_sc = [
        index
        for index, layer in enumerate(config["backbone"])
        if layer[2] == "C3k2_SCConv"
    ]
    yaml_head_sc = [
        backbone_length + index
        for index, layer in enumerate(config["head"])
        if layer[2] == "C3k2_SCConv"
    ]
    yaml_head_standard = sum(
        1 for layer in config["head"] if layer[2] == "C3k2"
    )
    yaml_stride2 = [
        index
        for index, layer in enumerate(static_layers)
        if layer[2] == "Conv" and len(layer[3]) >= 3 and layer[3][1:] == [3, 2]
    ]

    checks = {
        "only_requested_yaml_changes": config == expected_config,
        "backbone_scconv_count": yaml_backbone_sc == EXPECTED_SC_LAYERS
        and [node["layer"] for node in sc_nodes] == EXPECTED_SC_LAYERS,
        "neck_scconv_count": not yaml_head_sc
        and not any(node["region"] == "neck" for node in sc_nodes),
        "neck_standard_c3k2_count": yaml_head_standard == 4
        and sum(
            1
            for module in top_level[backbone_length:]
            if isinstance(module, C3k2) and not isinstance(module, C3k2_SCConv)
        )
        == 4,
        "stride2_conv_preserved": yaml_stride2 == EXPECTED_TOP_LEVEL_STRIDE2
        and stride2_layers == EXPECTED_TOP_LEVEL_STRIDE2,
        "detect_inputs_preserved": list(detect.f) == EXPECTED_DETECT_INPUTS
        and int(detect.nl) == 3,
        "cpu_forward_finite": args.device == "cpu" and forward_finite,
        "detect_feature_shapes": detect_feature_shapes == expected_feature_shapes,
    }
    report = {
        "model": str(model_path),
        "baseline_yaml": str(baseline_path),
        "backbone_length": backbone_length,
        "c3k2_scconv_nodes": sc_nodes,
        "backbone_c3k2_scconv": len(yaml_backbone_sc),
        "neck_c3k2_scconv": len(yaml_head_sc),
        "neck_standard_c3k2": yaml_head_standard,
        "top_level_stride2_conv_layers": stride2_layers,
        "detect_inputs": list(detect.f),
        "detect_levels": int(detect.nl),
        "detect_strides": [float(value) for value in detect.stride.cpu()],
        "forward_input_shape": list(sample.shape),
        "detect_feature_shapes": detect_feature_shapes,
        "forward_finite": forward_finite,
        "checks": {name: "PASS" if passed else "FAIL" for name, passed in checks.items()},
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
