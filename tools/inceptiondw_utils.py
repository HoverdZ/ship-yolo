"""Shared build, validation, statistics, and transfer helpers for InceptionDW."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "experiments" / "yolo11n_inceptiondw_c3k2_p23.yaml"
EXPERIMENT_NAME = "yolo11n_inceptiondw_c3k2_p23_640"
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
TARGET_LAYER_INDICES = (2, 4)
OFFICIAL_C3K2_INDICES = (6, 8, 13, 16, 19, 22)
DETECT_INDEX = 23


def require_ultralytics_version() -> str:
    """Fail early when the experiment is run with a different parser/model version."""

    import ultralytics

    version = getattr(ultralytics, "__version__", "unknown")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"This experiment targets ultralytics=={EXPECTED_ULTRALYTICS_VERSION}; found {version}."
        )
    return version


def register_modules() -> None:
    """Register repository-owned modules without editing site-packages."""

    require_ultralytics_version()
    from custom_modules.register import register_inceptiondw_modules

    register_inceptiondw_modules()


def build_custom_model(model_yaml: str | Path = MODEL_YAML):
    """Build the custom YOLO model after dynamic registration."""

    register_modules()
    from ultralytics import YOLO

    return YOLO(str(model_yaml))


def build_official_model():
    """Build the official YOLO11n architecture from the installed package."""

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


def model_statistics(yolo_model, imgsz: int = 640) -> dict[str, int | float]:
    """Return objective model size and compute statistics."""

    from ultralytics.utils.torch_utils import get_flops

    network = yolo_model.model
    parameters = sum(parameter.numel() for parameter in network.parameters())
    trainable = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
    leaf_layers = sum(1 for module in network.modules() if module is not network and not module._modules)
    gflops = float(get_flops(network, imgsz=imgsz))
    return {
        "layers": leaf_layers,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "gflops": gflops,
        "fp32_parameter_size_mib": parameters * 4 / (1024**2),
    }


def compare_statistics(
    baseline: dict[str, int | float],
    custom: dict[str, int | float],
) -> dict[str, dict[str, int | float | None]]:
    """Compute absolute and percentage changes from baseline."""

    comparison: dict[str, dict[str, int | float | None]] = {}
    for key, baseline_value in baseline.items():
        custom_value = custom[key]
        delta = custom_value - baseline_value
        percentage = (delta / baseline_value * 100.0) if baseline_value else None
        comparison[key] = {
            "baseline": baseline_value,
            "custom": custom_value,
            "delta": delta,
            "percent_change": percentage,
        }
    return comparison


def _module_prefix(state_key: str) -> str:
    parts = state_key.split(".")
    parameter_or_buffer_names = {
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
        "anchors",
        "strides",
    }
    while parts and parts[-1] in parameter_or_buffer_names:
        parts.pop()
    return ".".join(parts)


def _all_prefix_keys_matched(
    target_state: dict[str, torch.Tensor],
    matched_keys: set[str],
    prefix: str,
) -> bool:
    keys = [key for key in target_state if key.startswith(prefix)]
    return bool(keys) and all(key in matched_keys for key in keys)


def transfer_pretrained_weights(
    target_yolo,
    weights: str | Path = "yolo11n.pt",
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Report and optionally apply safe, exact-name, exact-shape weight transfer."""

    require_ultralytics_version()
    from ultralytics import YOLO

    source_yolo = YOLO(str(weights))
    source_state = source_yolo.model.float().state_dict()
    target_network = target_yolo.model
    target_state = target_network.state_dict()
    target_parameter_names = set(dict(target_network.named_parameters()))

    matched = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    matched_keys = set(matched)
    shape_mismatches = {
        key: {
            "source": list(value.shape),
            "target": list(target_state[key].shape),
        }
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
    }
    target_only = sorted(key for key in target_state if key not in matched_keys)
    source_only = sorted(key for key in source_state if key not in matched_keys)

    target_parameter_elements = sum(parameter.numel() for parameter in target_network.parameters())
    matched_parameter_elements = sum(
        target_state[key].numel() for key in matched_keys if key in target_parameter_names
    )
    target_state_elements = sum(value.numel() for value in target_state.values())
    matched_state_elements = sum(target_state[key].numel() for key in matched_keys)

    if apply:
        load_result = target_network.load_state_dict(matched, strict=False)
        unexpected_after_load = list(load_result.unexpected_keys)
    else:
        unexpected_after_load = []

    target_cv1 = {
        f"backbone_layer_{index}": _all_prefix_keys_matched(
            target_state,
            matched_keys,
            f"model.{index}.m.0.cv1.",
        )
        for index in TARGET_LAYER_INDICES
    }
    target_outer_1x1 = {
        f"backbone_layer_{index}": all(
            _all_prefix_keys_matched(target_state, matched_keys, f"model.{index}.{name}.")
            for name in ("cv1", "cv2")
        )
        for index in TARGET_LAYER_INDICES
    }
    neck_unmatched = sorted(
        key
        for key in target_only
        if any(key.startswith(f"model.{index}.") for index in range(11, 23))
    )
    detect_unmatched = sorted(
        key for key in target_only if key.startswith(f"model.{DETECT_INDEX}.")
    )
    untouched_backbone_unmatched = sorted(
        key
        for key in target_only
        if any(key.startswith(f"model.{index}.") for index in range(0, 11) if index not in TARGET_LAYER_INDICES)
    )
    replaced_source_conv_keys = sorted(
        key
        for key in source_state
        if any(key.startswith(f"model.{index}.m.0.cv2.conv.") for index in TARGET_LAYER_INDICES)
    )

    return {
        "weights": str(weights),
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
        "inherited_tensors": len(matched),
        "tensor_inheritance_ratio": len(matched) / len(target_state),
        "source_state_elements": sum(value.numel() for value in source_state.values()),
        "target_state_elements": target_state_elements,
        "inherited_state_elements": matched_state_elements,
        "state_element_inheritance_ratio": matched_state_elements / target_state_elements,
        "target_parameter_elements": target_parameter_elements,
        "inherited_parameter_elements": matched_parameter_elements,
        "parameter_element_inheritance_ratio": matched_parameter_elements / target_parameter_elements,
        "shape_mismatches": shape_mismatches,
        "unmatched_target_keys": target_only,
        "unmatched_source_keys": source_only,
        "unmatched_target_modules": sorted({_module_prefix(key) for key in target_only}),
        "p2_p3_cv1_inherited": target_cv1,
        "p2_p3_outer_1x1_inherited": target_outer_1x1,
        "untouched_backbone_unmatched": untouched_backbone_unmatched,
        "neck_unmatched": neck_unmatched,
        "detect_unmatched": detect_unmatched,
        "replaced_cv2_source_conv_keys": replaced_source_conv_keys,
        "replaced_cv2_source_conv_keys_inherited": any(key in matched_keys for key in replaced_source_conv_keys),
        "unexpected_after_load": unexpected_after_load,
        "applied": apply,
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


def forward_signature(yolo_model, imgsz: int = 640) -> dict[str, Any]:
    """Run one CPU forward and capture top-level and Detect-input shapes."""

    network = yolo_model.model.cpu().eval()
    top_level_shapes: dict[int, Any] = {}
    detect_inputs: list[list[int]] = []
    hooks = []

    for index, layer in enumerate(network.model[:-1]):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _args, output, layer_index=index: top_level_shapes.__setitem__(
                    layer_index,
                    _shape_tree(output),
                )
            )
        )

    def capture_detect_inputs(_module, args) -> None:
        features = args[0]
        detect_inputs.extend(list(feature.shape) for feature in features)

    hooks.append(network.model[DETECT_INDEX].register_forward_pre_hook(capture_detect_inputs))
    generator = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(1, 3, imgsz, imgsz, generator=generator)
    try:
        with torch.inference_mode():
            output = network(x)
    finally:
        for hook in hooks:
            hook.remove()

    output_tensors = list(_iter_tensors(output))
    return {
        "input_shape": list(x.shape),
        "detect_input_shapes": detect_inputs,
        "detection_spatial_sizes": [shape[-2:] for shape in detect_inputs],
        "top_level_shapes": {str(key): value for key, value in sorted(top_level_shapes.items())},
        "output_tree": _shape_tree(output),
        "output_tensor_count": len(output_tensors),
        "all_outputs_finite": bool(output_tensors) and all(
            bool(torch.isfinite(tensor).all()) for tensor in output_tensors
        ),
    }


def structure_report(custom_yolo) -> dict[str, Any]:
    """Inspect the exact target scope and reject unrelated modules."""

    from ultralytics.nn.modules import C3k2, Conv

    from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW, InceptionDWBottleneck
    from custom_modules.inceptiondw import InceptionDWConv2d, InceptionDWConvBNAct

    network = custom_yolo.model
    targets = [network.model[index] for index in TARGET_LAYER_INDICES]
    target_bottlenecks = [target.m[0] for target in targets]
    custom_indices = [
        index for index, layer in enumerate(network.model) if isinstance(layer, C3k2_InceptionDW)
    ]
    official_c3k2_indices = [
        index for index, layer in enumerate(network.model) if isinstance(layer, C3k2)
    ]
    inception_modules = [
        (name, module)
        for name, module in network.named_modules()
        if isinstance(module, InceptionDWConv2d)
    ]
    forbidden_modules = [
        f"{name}:{module.__class__.__name__}"
        for name, module in network.named_modules()
        if any(token in module.__class__.__name__.lower() for token in ("scconv", "sru", "cru"))
    ]
    wrapper_orders = {
        f"backbone_layer_{index}": list(bottleneck.cv2._modules)
        for index, bottleneck in zip(TARGET_LAYER_INDICES, target_bottlenecks)
    }

    checks = {
        "exactly_two_custom_c3k2": custom_indices == list(TARGET_LAYER_INDICES),
        "expected_inceptiondw_count": len(inception_modules) == 2,
        "one_scaled_repeat_per_target": all(len(target.m) == 1 for target in targets),
        "target_cv1_is_ultralytics_conv": all(
            isinstance(bottleneck.cv1, Conv) for bottleneck in target_bottlenecks
        ),
        "target_cv1_is_3x3": all(
            tuple(bottleneck.cv1.conv.kernel_size) == (3, 3) for bottleneck in target_bottlenecks
        ),
        "target_cv2_is_wrapper": all(
            isinstance(bottleneck.cv2, InceptionDWConvBNAct)
            for bottleneck in target_bottlenecks
        ),
        "target_bottleneck_type": all(
            isinstance(bottleneck, InceptionDWBottleneck)
            for bottleneck in target_bottlenecks
        ),
        "wrapper_order_is_inception_bn_silu": all(
            order == ["inception", "bn", "act"] for order in wrapper_orders.values()
        ),
        "residual_shortcut_preserved": all(bottleneck.add for bottleneck in target_bottlenecks),
        "p4_p5_and_neck_are_official_c3k2": all(
            isinstance(network.model[index], C3k2) and not isinstance(network.model[index], C3k2_InceptionDW)
            for index in OFFICIAL_C3K2_INDICES
        ),
        "official_c3k2_indices_unchanged": official_c3k2_indices == list(OFFICIAL_C3K2_INDICES),
        "no_scconv_sru_cru": not forbidden_modules,
        "detect_indices_unchanged": list(network.model[DETECT_INDEX].f) == [16, 19, 22],
    }

    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "custom_c3k2_indices": custom_indices,
        "official_c3k2_indices": official_c3k2_indices,
        "inceptiondw_modules": [name for name, _module in inception_modules],
        "inceptiondw_split_indexes": {
            name: list(module.split_indexes) for name, module in inception_modules
        },
        "wrapper_orders": wrapper_orders,
        "forbidden_modules": forbidden_modules,
    }


def yaml_scope_report(model_yaml: str | Path = MODEL_YAML) -> dict[str, Any]:
    """Compare experiment YAML against the installed official YOLO11 YAML."""

    custom = read_yaml(model_yaml)
    official_yolo = build_official_model()
    official = official_yolo.model.yaml
    custom_layers = custom["backbone"] + custom["head"]
    official_layers = official["backbone"] + official["head"]
    differences = []
    for index, (official_layer, custom_layer) in enumerate(zip(official_layers, custom_layers)):
        if official_layer != custom_layer:
            differences.append(
                {
                    "index": index,
                    "official": official_layer,
                    "custom": custom_layer,
                }
            )
    checks = {
        "same_layer_count": len(custom_layers) == len(official_layers),
        "only_layers_2_and_4_differ": [item["index"] for item in differences] == [2, 4],
        "only_module_name_changed": all(
            item["official"][:2] == item["custom"][:2]
            and item["official"][3] == item["custom"][3]
            and item["official"][2] == "C3k2"
            and item["custom"][2] == "C3k2_InceptionDW"
            for item in differences
        ),
        "scale_is_n": custom.get("scale") == "n",
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "differences": differences,
    }


def full_check(
    model_yaml: str | Path = MODEL_YAML,
    weights: str | Path = "yolo11n.pt",
    imgsz: int = 640,
) -> dict[str, Any]:
    """Run all non-training checks requested for the formal experiment."""

    custom = build_custom_model(model_yaml)
    official = build_official_model()
    scope = yaml_scope_report(model_yaml)
    structure = structure_report(custom)
    custom_forward = forward_signature(custom, imgsz=imgsz)
    official_forward = forward_signature(official, imgsz=imgsz)
    feature_shapes_match = (
        custom_forward["top_level_shapes"] == official_forward["top_level_shapes"]
        and custom_forward["detect_input_shapes"] == official_forward["detect_input_shapes"]
    )
    expected_spatial_sizes = [[imgsz // 8, imgsz // 8], [imgsz // 16, imgsz // 16], [imgsz // 32, imgsz // 32]]
    forward_checks = {
        "three_detection_scales": len(custom_forward["detect_input_shapes"]) == 3,
        "p3_p4_p5_spatial_sizes": custom_forward["detection_spatial_sizes"] == expected_spatial_sizes,
        "all_outputs_finite": custom_forward["all_outputs_finite"],
        "official_feature_shapes_match": feature_shapes_match,
    }
    baseline_stats = model_statistics(official, imgsz=imgsz)
    custom_stats = model_statistics(custom, imgsz=imgsz)
    transfer = transfer_pretrained_weights(custom, weights, apply=True)

    checks = {
        "yaml_scope": scope["all_checks_passed"],
        "model_structure": structure["all_checks_passed"],
        "forward": all(forward_checks.values()),
        "p2_p3_cv1_inherited": all(transfer["p2_p3_cv1_inherited"].values()),
        "p2_p3_outer_1x1_inherited": all(transfer["p2_p3_outer_1x1_inherited"].values()),
        "untouched_backbone_inherited": not transfer["untouched_backbone_unmatched"],
        "neck_inherited": not transfer["neck_unmatched"],
        "detect_inherited": not transfer["detect_unmatched"],
        "replaced_cv2_not_misloaded": not transfer["replaced_cv2_source_conv_keys_inherited"],
    }
    return {
        "experiment": EXPERIMENT_NAME,
        "ultralytics_version": require_ultralytics_version(),
        "model_yaml": str(Path(model_yaml).resolve()),
        "weights": str(weights),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "yaml_scope": scope,
        "structure": structure,
        "forward": {
            "checks": forward_checks,
            "custom": custom_forward,
            "official": official_forward,
        },
        "statistics": {
            "official_yolo11n": baseline_stats,
            "inceptiondw": custom_stats,
            "comparison": compare_statistics(baseline_stats, custom_stats),
        },
        "weight_transfer": transfer,
    }
