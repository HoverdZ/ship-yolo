"""Build, structure, transfer, and CPU-forward helpers for module ablations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from custom_modules.c3k2_crossconv import C3k2CrossConv
from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
from custom_modules.cgfm import AlignConcat, CGFM
from custom_modules.dd import DD
from custom_modules.register import register_module_ablation_modules

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "yolo11n-c3cross": ROOT / "experiments/yolo11n-c3cross.yaml",
    "yolo11n-dd": ROOT / "experiments/yolo11n-dd.yaml",
    "yolo11n-cgfm": ROOT / "experiments/yolo11n-cgfm.yaml",
    "yolo11n-inceptiondw-dd": ROOT / "experiments/yolo11n-inceptiondw-dd.yaml",
    "yolo11n-inceptiondw-cgfm": ROOT / "experiments/yolo11n-inceptiondw-cgfm.yaml",
}
CONTROL_EXPERIMENT = {
    "yolo11n-alignconcat-control": ROOT / "experiments/yolo11n-alignconcat-control.yaml",
}


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_model(model_yaml: str | Path) -> YOLO:
    """Build a registered model from YAML without starting training."""
    register_module_ablation_modules()
    return YOLO(str(model_yaml), verbose=False)


def model_statistics(model: YOLO, imgsz: int = 640) -> dict[str, int | float]:
    network = model.model
    return {
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in network.parameters() if parameter.requires_grad
        ),
        "gflops": float(get_flops(network, imgsz=imgsz)),
    }


def transfer_pretrained_weights(
    target: YOLO,
    weights: str | Path,
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Apply only exact-name, exact-shape official checkpoint tensors."""
    source = YOLO(str(weights), verbose=False).model.float().state_dict()
    target_network = target.model
    target_state = target_network.state_dict()
    parameter_names = set(dict(target_network.named_parameters()))
    matched = {
        key: tensor
        for key, tensor in source.items()
        if key in target_state and tuple(tensor.shape) == tuple(target_state[key].shape)
    }
    matched_keys = set(matched)
    if apply:
        load_result = target_network.load_state_dict(matched, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {load_result.unexpected_keys}")

    target_parameter_elements = sum(parameter.numel() for parameter in target_network.parameters())
    matched_parameter_elements = sum(
        target_state[key].numel() for key in matched_keys if key in parameter_names
    )
    unmatched_target = sorted(key for key in target_state if key not in matched_keys)
    return {
        "weights": _portable_path(weights),
        "source_tensors": len(source),
        "target_tensors": len(target_state),
        "matched_tensors": len(matched),
        "unmatched_target_tensors": len(unmatched_target),
        "tensor_match_ratio": len(matched) / len(target_state),
        "target_parameter_elements": target_parameter_elements,
        "matched_parameter_elements": matched_parameter_elements,
        "parameter_element_match_ratio": (
            matched_parameter_elements / target_parameter_elements
        ),
        "unmatched_target_keys": unmatched_target,
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


def cpu_forward_report(model: YOLO, imgsz: int = 640) -> dict[str, Any]:
    """Run a CPU forward and capture the three features entering Detect."""
    network = model.model.cpu().eval()
    detect_inputs: list[list[int]] = []

    def capture_detect_inputs(_module, args) -> None:
        features = args[0]
        detect_inputs.extend([list(feature.shape) for feature in features])

    handle = network.model[-1].register_forward_pre_hook(capture_detect_inputs)
    generator = torch.Generator(device="cpu").manual_seed(0)
    image = torch.randn(1, 3, imgsz, imgsz, generator=generator)
    try:
        with torch.inference_mode():
            output = network(image)
    finally:
        handle.remove()
    tensors = list(_iter_tensors(output))
    return {
        "input_shape": list(image.shape),
        "output_shapes": _shape_tree(output),
        "detect_input_shapes": detect_inputs,
        "detect_input_channels": [shape[1] for shape in detect_inputs],
        "all_outputs_finite": all(torch.isfinite(tensor).all().item() for tensor in tensors),
        "passed": len(detect_inputs) == 3 and all(
            torch.isfinite(tensor).all().item() for tensor in tensors
        ),
    }


def structure_report(model: YOLO, experiment_name: str) -> dict[str, Any]:
    """Check exact replacement locations and preserve a three-scale Detect head."""
    layers = model.model.model
    cross_indices = [i for i, layer in enumerate(layers) if isinstance(layer, C3k2CrossConv)]
    dd_indices = [i for i, layer in enumerate(layers) if isinstance(layer, DD)]
    cgfm_indices = [i for i, layer in enumerate(layers) if isinstance(layer, CGFM)]
    align_indices = [i for i, layer in enumerate(layers) if isinstance(layer, AlignConcat)]
    inception_indices = [i for i, layer in enumerate(layers) if isinstance(layer, C3k2_InceptionDW)]

    expected = {
        "yolo11n-c3cross": ([2, 4, 6, 8], [], [], [], []),
        "yolo11n-dd": ([], [1, 3, 5, 7], [], [], []),
        "yolo11n-cgfm": ([], [], [12], [], []),
        "yolo11n-inceptiondw-dd": ([], [1, 3, 5, 7], [], [], [2, 4]),
        "yolo11n-inceptiondw-cgfm": ([], [], [12], [], [2, 4]),
        "yolo11n-alignconcat-control": ([], [], [], [12], []),
    }[experiment_name]
    actual = (cross_indices, dd_indices, cgfm_indices, align_indices, inception_indices)
    detect_from = list(layers[-1].f)
    return {
        "crossconv_indices": cross_indices,
        "dd_indices": dd_indices,
        "cgfm_indices": cgfm_indices,
        "alignconcat_indices": align_indices,
        "inceptiondw_indices": inception_indices,
        "detect_from": detect_from,
        "three_scale_detect": detect_from == [16, 19, 22],
        "replacement_scope_passed": actual == expected,
        "passed": actual == expected and detect_from == [16, 19, 22],
    }


def validate_experiment(
    experiment_name: str,
    model_yaml: str | Path,
    weights: str | Path,
    imgsz: int = 640,
) -> dict[str, Any]:
    """Run non-training validation for one experiment."""
    model = build_model(model_yaml)
    structure = structure_report(model, experiment_name)
    statistics = model_statistics(model, imgsz=imgsz)
    transfer = transfer_pretrained_weights(model, weights, apply=True)
    forward = cpu_forward_report(model, imgsz=imgsz)
    return {
        "experiment": experiment_name,
        "yaml": _portable_path(model_yaml),
        "statistics": statistics,
        "structure": structure,
        "transfer": transfer,
        "cpu_forward": forward,
        "passed": structure["passed"] and forward["passed"],
    }
