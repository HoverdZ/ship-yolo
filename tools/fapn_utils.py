"""Shared model, topology, validation, and weight-transfer utilities for FaPN."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torchvision.ops import DeformConv2d


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
VARIANTS = {
    "baseline": {
        "name": "yolo11n_fapn_640",
        "yaml": ROOT / "experiments" / "yolo11n_fapn.yaml",
        "report": ROOT / "artifacts" / "fapn_weight_transfer_baseline.json",
        "inceptiondw": False,
    },
    "inceptiondw": {
        "name": "yolo11n_inceptiondw_fapn_640",
        "yaml": ROOT / "experiments" / "yolo11n_inceptiondw_fapn.yaml",
        "report": ROOT / "artifacts" / "fapn_weight_transfer_inceptiondw.json",
        "inceptiondw": True,
    },
}

FAPN_INDICES = (11, 12, 13, 14, 15)
FAPN_ALIGN_INDICES = (12, 14)
FAPN_OUTPUT_INDICES = (13, 15)
PAN_INDICES = (16, 17, 18, 19, 20, 21)
DETECT_INDEX = 22

# Explicit semantic mapping from official YOLO11 source layers to the shifted
# FaPN target. Top-down source layers 11-16 are intentionally excluded.
SOURCE_TO_TARGET_LAYER_MAP = {
    **{index: index for index in range(11)},
    17: 16,  # first PAN stride-2 Conv
    19: 18,  # first PAN output C3k2
    20: 19,  # second PAN stride-2 Conv
    22: 21,  # second PAN output C3k2
    23: 22,  # Detect
}
TARGET_TO_SOURCE_LAYER_MAP = {target: source for source, target in SOURCE_TO_TARGET_LAYER_MAP.items()}


def require_ultralytics_version() -> str:
    import ultralytics

    version = getattr(ultralytics, "__version__", "unknown")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"FaPN experiments target ultralytics=={EXPECTED_ULTRALYTICS_VERSION}; found {version}."
        )
    return version


def register_modules() -> None:
    require_ultralytics_version()
    from custom_modules.register import register_fapn_modules

    register_fapn_modules()


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown FaPN variant {variant!r}; expected one of {sorted(VARIANTS)}.")
    return VARIANTS[variant]


def build_model(variant: str):
    register_modules()
    from ultralytics import YOLO

    return YOLO(str(variant_config(variant)["yaml"]))


def build_official_model():
    require_ultralytics_version()
    from ultralytics import YOLO

    return YOLO("yolo11n.yaml")


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _deform_conv_macs(module: DeformConv2d, inputs, output: torch.Tensor) -> None:
    """THOP hook: count DCNv2 convolution MACs as the equivalent 3x3 convolution."""

    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
    macs = output.numel() * kernel_ops
    module.total_ops += torch.DoubleTensor([macs])


def model_statistics(yolo_model, imgsz: int = 640) -> dict[str, int | float]:
    """Report parameters and THOP GFLOPs with an explicit DeformConv2d hook."""

    from thop import profile

    network = yolo_model.model.cpu().eval()
    parameters = sum(parameter.numel() for parameter in network.parameters())
    trainable = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
    layers = sum(1 for module in network.modules() if module is not network and not module._modules)
    sample = torch.zeros(1, 3, imgsz, imgsz)
    with torch.inference_mode():
        macs, _ = profile(
            deepcopy(network),
            inputs=(sample,),
            custom_ops={DeformConv2d: _deform_conv_macs},
            verbose=False,
        )
    return {
        "layers": layers,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "gflops": float(macs * 2 / 1e9),
        "fp32_parameter_size_mib": parameters * 4 / (1024**2),
        "gflops_note": "THOP with an explicit DCNv2-as-3x3-convolution MAC hook.",
    }


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _shape_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_tree(item) for item in value]
    return type(value).__name__


def topology_report() -> dict[str, Any]:
    """Verify YAML-level ablation scope before constructing the models."""

    baseline = read_yaml(VARIANTS["baseline"]["yaml"])
    inception = read_yaml(VARIANTS["inceptiondw"]["yaml"])
    validated_inception = read_yaml(ROOT / "experiments" / "yolo11n_inceptiondw_c3k2_p23.yaml")
    official = build_official_model().model.yaml

    baseline_layers = baseline["backbone"] + baseline["head"]
    inception_layers = inception["backbone"] + inception["head"]
    checks = {
        "both_nc_equal_one": baseline["nc"] == inception["nc"] == 1,
        "baseline_backbone_is_official": baseline["backbone"] == official["backbone"],
        "inception_backbone_is_validated_version": inception["backbone"] == validated_inception["backbone"],
        "both_fapn_pan_detect_heads_identical": baseline["head"] == inception["head"],
        "only_backbone_layers_2_and_4_differ_between_variants": [
            index
            for index, (left, right) in enumerate(zip(baseline_layers, inception_layers))
            if left != right
        ]
        == [2, 4],
        "m4_recurses_not_t4": baseline_layers[14][0] == [4, 12],
        "pan_second_concat_uses_original_c5": baseline_layers[20][0] == [-1, 10],
        "detect_uses_p3_p4_p5": baseline_layers[22][0] == [15, 18, 21],
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "combined_layer_count": len(baseline_layers),
        "fapn_indices": list(FAPN_INDICES),
        "pan_indices": list(PAN_INDICES),
        "detect_index": DETECT_INDEX,
    }


def structure_report(yolo_model, variant: str) -> dict[str, Any]:
    """Inspect module counts, placement, channels, DCNv2 rules, and PAN scope."""

    from ultralytics.nn.modules import C3k2, Concat, Conv, Detect

    from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
    from custom_modules.fapn import (
        FaPNAlign,
        FaPNFeatureSelection,
        FaPNLateral,
        FaPNModulatedDeformConv2d,
        FaPNOutputConv,
    )

    network = yolo_model.model
    modules = list(network.modules())
    aligns = [module for module in modules if isinstance(module, FaPNAlign)]
    fsms = [module for module in modules if isinstance(module, FaPNFeatureSelection)]
    dcns = [module for module in modules if isinstance(module, FaPNModulatedDeformConv2d)]
    outputs = [module for module in modules if isinstance(module, FaPNOutputConv)]
    laterals = [module for module in modules if isinstance(module, FaPNLateral)]
    inception = [name for name, module in network.named_modules() if isinstance(module, C3k2_InceptionDW)]
    deform_paths = [name for name, module in network.named_modules() if isinstance(module, DeformConv2d)]
    forbidden = [
        f"{name}:{module.__class__.__name__}"
        for name, module in network.named_modules()
        if any(token in module.__class__.__name__.lower() for token in ("dysample", "dcnv3", "dcnv4"))
    ]

    pan_stride_convs = [
        index
        for index in (16, 19)
        if isinstance(network.model[index], Conv) and tuple(network.model[index].conv.stride) == (2, 2)
    ]
    pan_concats = [index for index in (17, 20) if isinstance(network.model[index], Concat)]
    pan_c3k2 = [
        index
        for index in (18, 21)
        if isinstance(network.model[index], C3k2) and not isinstance(network.model[index], C3k2_InceptionDW)
    ]
    offset_initialization = [
        bool(torch.count_nonzero(module.conv_offset_mask.weight) == 0)
        and bool(torch.count_nonzero(module.conv_offset_mask.bias) == 0)
        for module in dcns
    ]
    channel_rules = [
        {
            "channels": module.channels,
            "divisible_by_8": module.channels % 8 == 0,
            "deformable_groups": module.deformable_groups,
            "offset_channels": module.offset_channels,
            "mask_channels": module.mask_channels,
            "expected_offset_channels": 2 * 8 * 3 * 3,
            "expected_mask_channels": 8 * 3 * 3,
        }
        for module in dcns
    ]
    checks = {
        "two_fapn_align": len(aligns) == 2,
        "two_fsm": len(fsms) == 2,
        "two_modulated_dcnv2": len(dcns) == 2,
        "two_output_convs": len(outputs) == 2,
        "one_c5_lateral": len(laterals) == 1,
        "pan_two_stride_convs": pan_stride_convs == [16, 19],
        "pan_two_concats": pan_concats == [17, 20],
        "pan_two_original_output_c3k2": pan_c3k2 == [18, 21],
        "pan_has_no_dcn": all(path.startswith(("model.12.", "model.14.")) for path in deform_paths),
        "detect_is_three_scale": isinstance(network.model[DETECT_INDEX], Detect)
        and list(network.model[DETECT_INDEX].f) == [15, 18, 21],
        "no_forbidden_modules": not forbidden,
        "offset_mask_zero_initialized": len(offset_initialization) == 2 and all(offset_initialization),
        "dcn_channel_rules": len(channel_rules) == 2
        and all(
            item["divisible_by_8"]
            and item["deformable_groups"] == 8
            and item["offset_channels"] == item["expected_offset_channels"]
            and item["mask_channels"] == item["expected_mask_channels"]
            for item in channel_rules
        ),
        "inception_scope": len(inception) == (2 if variant == "inceptiondw" else 0),
    }
    return {
        "variant": variant,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "counts": {
            "FaPNAlign": len(aligns),
            "FaPNFeatureSelection": len(fsms),
            "FaPNModulatedDeformConv2d": len(dcns),
            "FaPNOutputConv": len(outputs),
            "FaPNLateral": len(laterals),
        },
        "pan": {
            "stride_conv_indices": pan_stride_convs,
            "concat_indices": pan_concats,
            "c3k2_indices": pan_c3k2,
            "second_concat_from": list(network.model[20].f),
        },
        "inceptiondw_modules": inception,
        "deform_conv_paths": deform_paths,
        "channel_rules": channel_rules,
        "forbidden_modules": forbidden,
    }


def forward_report(yolo_model, imgsz: int = 640) -> dict[str, Any]:
    """Run CPU inference and capture every FaPN node plus Detect input shapes."""

    network = yolo_model.model.cpu().eval()
    node_shapes: dict[str, dict[str, Any]] = {}
    detect_inputs: list[list[int]] = []
    hooks = []

    for index in FAPN_INDICES:
        layer = network.model[index]
        hooks.append(
            layer.register_forward_hook(
                lambda _module, inputs, output, layer_index=index: node_shapes.__setitem__(
                    str(layer_index),
                    {"input": _shape_tree(inputs[0]), "output": _shape_tree(output)},
                )
            )
        )

    def capture_detect(_module, inputs) -> None:
        detect_inputs.extend(list(feature.shape) for feature in inputs[0])

    hooks.append(network.model[DETECT_INDEX].register_forward_pre_hook(capture_detect))
    generator = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(1, 3, imgsz, imgsz, generator=generator)
    try:
        with torch.inference_mode():
            output = network(x)
    finally:
        for hook in hooks:
            hook.remove()

    tensors = list(_iter_tensors(output))
    expected_sizes = [[imgsz // 8, imgsz // 8], [imgsz // 16, imgsz // 16], [imgsz // 32, imgsz // 32]]
    checks = {
        "three_detect_inputs": len(detect_inputs) == 3,
        "p3_p4_p5_strides": [shape[-2:] for shape in detect_inputs] == expected_sizes,
        "all_outputs_finite": bool(tensors) and all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "input_shape": list(x.shape),
        "fapn_nodes": node_shapes,
        "detect_input_shapes": detect_inputs,
        "output_shape_tree": _shape_tree(output),
    }


def backward_report(yolo_model, imgsz: int = 256) -> dict[str, Any]:
    """Run a small differentiable CPU pass and audit all FaPN parameter gradients."""

    network = yolo_model.model.cpu().train()
    network.zero_grad(set_to_none=True)
    generator = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(1, 3, imgsz, imgsz, generator=generator, requires_grad=True)
    output = network(x)
    tensors = list(_iter_tensors(output))
    if not tensors:
        raise RuntimeError("Model forward returned no tensors for backward validation.")
    loss = sum(tensor.float().square().mean() for tensor in tensors)
    loss.backward()

    fapn_parameters = {
        name: parameter
        for name, parameter in network.named_parameters()
        if any(name.startswith(f"model.{index}.") for index in FAPN_INDICES) and parameter.requires_grad
    }
    missing = sorted(name for name, parameter in fapn_parameters.items() if parameter.grad is None)
    nonfinite = sorted(
        name
        for name, parameter in fapn_parameters.items()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    )
    zero_grad = sorted(
        name
        for name, parameter in fapn_parameters.items()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) == 0)
    )
    expected_zero_grad = {
        "model.12.offset_feature.weight",
        "model.14.offset_feature.weight",
    }
    unexpected_zero_grad = sorted(set(zero_grad) - expected_zero_grad)
    checks = {
        "loss_finite": bool(torch.isfinite(loss)),
        "input_gradient_present": x.grad is not None and bool(torch.isfinite(x.grad).all()),
        "all_fapn_parameters_have_gradients": not missing,
        "all_fapn_gradients_finite": not nonfinite,
        # At official zero offset/mask-conv initialization, its upstream
        # offset-feature 1x1 receives a mathematically valid zero gradient on
        # the first step. It becomes trainable as soon as the zero-initialized
        # generator updates. No other FaPN parameter may have a zero gradient.
        "zero_gradients_are_official_initialization_effect": not unexpected_zero_grad,
    }
    network.zero_grad(set_to_none=True)
    network.eval()
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "input_shape": list(x.shape),
        "loss": float(loss.detach()),
        "fapn_parameter_tensors": len(fapn_parameters),
        "missing_gradient_keys": missing,
        "nonfinite_gradient_keys": nonfinite,
        "zero_gradient_keys": zero_grad,
        "unexpected_zero_gradient_keys": unexpected_zero_grad,
    }


def _state_layer_index(key: str) -> int | None:
    match = re.match(r"model\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def _replace_layer_index(key: str, source_index: int) -> str:
    return re.sub(r"^model\.\d+\.", f"model.{source_index}.", key, count=1)


def _category_for_target_key(key: str) -> str:
    index = _state_layer_index(key)
    if index is None:
        return "other"
    if 0 <= index <= 10:
        return "backbone"
    if index in FAPN_INDICES:
        return "fapn"
    if index in PAN_INDICES:
        return "pan"
    if index == DETECT_INDEX:
        return "detect"
    return "other"


def semantic_weight_transfer(
    target_yolo,
    weights: str | Path = "yolo11n.pt",
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Map official weights by semantic layer identity, never shifted raw names."""

    require_ultralytics_version()
    from ultralytics import YOLO

    source_yolo = YOLO(str(weights))
    source_state = source_yolo.model.float().state_dict()
    target_network = target_yolo.model
    target_state = target_network.state_dict()
    target_parameters = dict(target_network.named_parameters())

    inherited: dict[str, torch.Tensor] = {}
    target_to_source_keys: dict[str, str] = {}
    shape_mismatches: list[dict[str, Any]] = []
    unmapped_target: list[str] = []

    for target_key, target_value in target_state.items():
        target_index = _state_layer_index(target_key)
        source_index = TARGET_TO_SOURCE_LAYER_MAP.get(target_index) if target_index is not None else None
        if source_index is None:
            unmapped_target.append(target_key)
            continue
        source_key = _replace_layer_index(target_key, source_index)
        source_value = source_state.get(source_key)
        if source_value is None:
            unmapped_target.append(target_key)
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            shape_mismatches.append(
                {
                    "target_key": target_key,
                    "source_key": source_key,
                    "target_shape": list(target_value.shape),
                    "source_shape": list(source_value.shape),
                    "category": _category_for_target_key(target_key),
                }
            )
            continue
        inherited[target_key] = source_value
        target_to_source_keys[target_key] = source_key

    if apply:
        load_result = target_network.load_state_dict(inherited, strict=False)
        unexpected_after_load = list(load_result.unexpected_keys)
    else:
        unexpected_after_load = []

    inherited_keys = set(inherited)
    used_source_keys = set(target_to_source_keys.values())
    unmatched_target_keys = sorted(key for key in target_state if key not in inherited_keys)
    unmatched_source_keys = sorted(key for key in source_state if key not in used_source_keys)

    category_report: dict[str, dict[str, int | float]] = {}
    for category in ("backbone", "fapn", "pan", "detect", "other"):
        parameter_keys = [key for key in target_parameters if _category_for_target_key(key) == category]
        total_tensors = len(parameter_keys)
        inherited_tensors = sum(key in inherited_keys for key in parameter_keys)
        total_elements = sum(target_parameters[key].numel() for key in parameter_keys)
        inherited_elements = sum(
            target_parameters[key].numel() for key in parameter_keys if key in inherited_keys
        )
        category_report[category] = {
            "target_parameter_tensors": total_tensors,
            "inherited_parameter_tensors": inherited_tensors,
            "tensor_ratio": inherited_tensors / total_tensors if total_tensors else 1.0,
            "target_parameter_elements": total_elements,
            "inherited_parameter_elements": inherited_elements,
            "element_ratio": inherited_elements / total_elements if total_elements else 1.0,
        }

    target_parameter_elements = sum(parameter.numel() for parameter in target_parameters.values())
    inherited_parameter_elements = sum(
        target_parameters[key].numel() for key in inherited_keys if key in target_parameters
    )
    target_parameter_tensors = len(target_parameters)
    inherited_parameter_tensors = sum(key in target_parameters for key in inherited_keys)

    return {
        "weights": str(weights),
        "mapping_strategy": "explicit semantic layer mapping",
        "source_to_target_layer_map": {str(key): value for key, value in SOURCE_TO_TARGET_LAYER_MAP.items()},
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
        "inherited_state_tensors": len(inherited),
        "target_parameter_tensors": target_parameter_tensors,
        "inherited_parameter_tensors": inherited_parameter_tensors,
        "target_parameter_elements": target_parameter_elements,
        "inherited_parameter_elements": inherited_parameter_elements,
        "randomly_initialized_parameter_elements": target_parameter_elements - inherited_parameter_elements,
        "parameter_element_inheritance_ratio": inherited_parameter_elements / target_parameter_elements,
        "category_report": category_report,
        "shape_mismatches": shape_mismatches,
        "unmatched_target_keys": unmatched_target_keys,
        "unmatched_source_keys": unmatched_source_keys,
        "target_to_source_keys": target_to_source_keys,
        "unexpected_after_load": unexpected_after_load,
        "applied": apply,
    }


def full_check(variant: str, weights: str | Path = "yolo11n.pt", imgsz: int = 640) -> dict[str, Any]:
    """Run every local non-training verification for one FaPN variant."""

    model = build_model(variant)
    transfer = semantic_weight_transfer(model, weights, apply=True)
    topology = topology_report()
    structure = structure_report(model, variant)
    forward = forward_report(model, imgsz=imgsz)
    backward = backward_report(model, imgsz=256)
    stats = model_statistics(model, imgsz=imgsz)
    checks = {
        "topology": topology["all_checks_passed"],
        "structure": structure["all_checks_passed"],
        "forward": forward["all_checks_passed"],
        "backward": backward["all_checks_passed"],
    }
    return {
        "variant": variant,
        "experiment_name": variant_config(variant)["name"],
        "model_yaml": str(variant_config(variant)["yaml"]),
        "ultralytics_version": require_ultralytics_version(),
        "python_version": sys.version.split()[0],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "topology": topology,
        "structure": structure,
        "forward": forward,
        "backward": backward,
        "statistics": stats,
        "weight_transfer": transfer,
    }
