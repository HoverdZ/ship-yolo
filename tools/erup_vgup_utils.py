"""Construction, initialization, and audit helpers for ERUP/VGUP models."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
EXPECTED_STRIDES = [4.0, 8.0, 16.0]

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "incdw_dysample_sfl_scam_vgup": {
        "yaml": ROOT
        / "experiments/yolo11n_incdw_dysample_sfl_scam_vgup.yaml",
        "base": "incdw_dysample_pls_scam",
        "preprocessor": "vgup",
        "uses_scam": True,
        "detect_index": 25,
        "detect_from": [22, 23, 24],
    },
    "incdw_dysample_sfl_vgup": {
        "yaml": ROOT
        / "experiments/yolo11n_incdw_dysample_sfl_vgup.yaml",
        "base": "incdw_dysample_pls",
        "preprocessor": "vgup",
        "uses_scam": False,
        "detect_index": 22,
        "detect_from": [15, 18, 21],
    },
    "incdw_dysample_sfl_scam_erup": {
        "yaml": ROOT
        / "experiments/yolo11n_incdw_dysample_sfl_scam_erup.yaml",
        "base": "incdw_dysample_pls_scam",
        "preprocessor": "erup",
        "uses_scam": True,
        "detect_index": 25,
        "detect_from": [22, 23, 24],
    },
    "incdw_dysample_sfl_erup": {
        "yaml": ROOT
        / "experiments/yolo11n_incdw_dysample_sfl_erup.yaml",
        "base": "incdw_dysample_pls",
        "preprocessor": "erup",
        "uses_scam": False,
        "detect_index": 22,
        "detect_from": [15, 18, 21],
    },
}


def require_ultralytics_version() -> str:
    import ultralytics

    version = getattr(ultralytics, "__version__", "unknown")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected ultralytics=={EXPECTED_ULTRALYTICS_VERSION}, found {version}."
        )
    return version


def register_modules() -> None:
    require_ultralytics_version()
    from custom_modules.register import register_adaptive_preprocessors

    register_adaptive_preprocessors()


def build_model(
    experiment: str,
    *,
    nc: int | None = None,
    verbose: bool = False,
):
    if experiment not in EXPERIMENTS:
        raise KeyError(f"Unknown ERUP/VGUP experiment: {experiment}")
    register_modules()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    model_yaml = EXPERIMENTS[experiment]["yaml"]
    wrapper = YOLO(str(model_yaml), verbose=verbose)
    current_nc = int(wrapper.model.model[-1].nc)
    if nc is not None and current_nc != int(nc):
        wrapper.model = DetectionModel(
            cfg=str(model_yaml),
            ch=3,
            nc=int(nc),
            verbose=verbose,
        )
        wrapper.task = "detect"
        wrapper.ckpt = wrapper.ckpt or {}
    return wrapper


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def model_statistics(model, *, imgsz: int = 640) -> dict[str, Any]:
    from ultralytics.utils.torch_utils import get_flops

    network = model.model
    preprocessor = network.model[0]
    return {
        "parameters": sum(item.numel() for item in network.parameters()),
        "trainable_parameters": sum(
            item.numel() for item in network.parameters() if item.requires_grad
        ),
        "preprocessor_parameters": sum(
            item.numel() for item in preprocessor.parameters()
        ),
        "detector_parameters": sum(
            item.numel()
            for layer in network.model[1:]
            for item in layer.parameters()
        ),
        "state_tensors": len(network.state_dict()),
        "gflops": float(get_flops(network, imgsz=imgsz)),
        "gflops_note": (
            "THOP may undercount BPW piecewise arithmetic and per-sample "
            "dynamic KBL convolution."
        ),
    }


def _shift_source_key(target_key: str) -> str | None:
    parts = target_key.split(".")
    if len(parts) < 3 or parts[0] != "model":
        return None
    try:
        layer = int(parts[1])
    except ValueError:
        return None
    if layer == 0:
        return None
    parts[1] = str(layer - 1)
    return ".".join(parts)


def initialize_from_official(
    target,
    experiment: str,
    *,
    weights: str | Path = "yolo11n.pt",
    apply: bool,
) -> dict[str, Any]:
    """Initialize the matching base detector, then copy it after layer shift."""

    from tools.cumulative_models_utils import (
        build_model as build_base_model,
        transfer_pretrained_weights,
    )

    spec = EXPERIMENTS[experiment]
    target_nc = int(target.model.model[-1].nc)
    torch.manual_seed(0)
    base = build_base_model(spec["base"], nc=target_nc)
    official = transfer_pretrained_weights(
        base,
        weights,
        apply=True,
    )
    if not official["passed"]:
        raise RuntimeError(
            "Base detector official-weight inheritance failed: "
            f"{official['verification_failures']}"
        )

    source_state = base.model.state_dict()
    target_state = target.model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    mapping: dict[str, str] = {}
    preprocessor_keys: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for target_key, target_value in target_state.items():
        source_key = _shift_source_key(target_key)
        if source_key is None:
            preprocessor_keys.append(target_key)
            continue
        source_value = source_state.get(source_key)
        if source_value is None:
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            shape_mismatches.append(
                {
                    "target": target_key,
                    "source": source_key,
                    "target_shape": list(target_value.shape),
                    "source_shape": list(source_value.shape),
                }
            )
            continue
        mapped[target_key] = source_value.detach().cpu()
        mapping[target_key] = source_key

    missing_after_load: list[str] = []
    unexpected_after_load: list[str] = []
    verification_failures: list[str] = []
    if apply:
        result = target.model.load_state_dict(mapped, strict=False)
        missing_after_load = list(result.missing_keys)
        unexpected_after_load = list(result.unexpected_keys)
        loaded = target.model.state_dict()
        verification_failures = [
            key
            for key, expected in mapped.items()
            if not torch.equal(loaded[key].detach().cpu(), expected)
        ]

    detector_target_keys = [
        key for key in target_state if _shift_source_key(key) is not None
    ]
    detector_parameter_names = {
        key
        for key, _value in target.model.named_parameters()
        if _shift_source_key(key) is not None
    }
    detector_parameter_elements = sum(
        target_state[key].numel() for key in detector_parameter_names
    )
    copied_parameter_elements = sum(
        target_state[key].numel()
        for key in mapped
        if key in detector_parameter_names
    )
    reported_missing = set(target_state) - set(mapped)
    passed = (
        not shape_mismatches
        and len(mapped) == len(detector_target_keys)
        and (
            not apply
            or (
                set(missing_after_load).issubset(reported_missing)
                and not unexpected_after_load
                and not verification_failures
            )
        )
    )
    return {
        "experiment": experiment,
        "base_experiment": spec["base"],
        "official_weights": str(weights),
        "official_to_base": official,
        "target_state_tensors": len(target_state),
        "preprocessor_state_tensors": len(preprocessor_keys),
        "detector_state_tensors": len(detector_target_keys),
        "detector_loaded_tensors": len(mapped),
        "detector_tensor_inheritance_ratio": (
            len(mapped) / len(detector_target_keys)
        ),
        "detector_parameter_elements": detector_parameter_elements,
        "detector_loaded_parameter_elements": copied_parameter_elements,
        "detector_parameter_element_ratio": (
            copied_parameter_elements / detector_parameter_elements
        ),
        "preprocessor_random_keys": sorted(preprocessor_keys),
        "shape_mismatches": shape_mismatches,
        "missing_after_load": missing_after_load,
        "unexpected_after_load": unexpected_after_load,
        "verification_failures": verification_failures,
        "sample_mapping": dict(list(sorted(mapping.items()))[:20]),
        "applied": apply,
        "passed": passed,
    }


def structure_report(model, experiment: str) -> dict[str, Any]:
    from custom_modules.erup import ERUPPreprocessor
    from custom_modules.scam import SCAM
    from custom_modules.vgup import VGUPPreprocessor
    from tools.cumulative_models_utils import build_model as build_base_model

    spec = EXPERIMENTS[experiment]
    network = model.model
    layers = network.model
    base = build_base_model(spec["base"])
    base_layers = base.model.model
    preprocessor = layers[0]
    expected_type = (
        VGUPPreprocessor
        if spec["preprocessor"] == "vgup"
        else ERUPPreprocessor
    )
    scam_count = sum(isinstance(layer, SCAM) for layer in layers)
    detector = layers[spec["detect_index"]]
    layer_types_match = len(layers) == len(base_layers) + 1 and all(
        type(adaptive) is type(original)
        for adaptive, original in zip(layers[1:], base_layers, strict=True)
    )
    checks = {
        "preprocessor_is_first": isinstance(preprocessor, expected_type),
        "detector_layer_types_unchanged": layer_types_match,
        "detect_from": list(detector.f) == spec["detect_from"],
        "strides": [float(item) for item in network.stride]
        == EXPECTED_STRIDES,
        "scam_presence": scam_count == (3 if spec["uses_scam"] else 0),
    }
    return {
        "preprocessor": type(preprocessor).__name__,
        "detect_index": spec["detect_index"],
        "detect_from": list(detector.f),
        "strides": [float(item) for item in network.stride],
        "scam_count": scam_count,
        "base_experiment": spec["base"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def forward_report(
    model,
    experiment: str,
    *,
    imgsz: int = 640,
) -> dict[str, Any]:
    spec = EXPERIMENTS[experiment]
    network = model.model.cpu().eval()
    preprocessor_shapes: list[dict[str, list[int]]] = []
    detect_inputs: list[list[int]] = []

    def capture_preprocessor(_module, args, output) -> None:
        preprocessor_shapes.append(
            {
                "input": list(args[0].shape),
                "output": list(output.shape),
            }
        )

    def capture_detect(_module, args) -> None:
        detect_inputs.extend(list(item.shape) for item in args[0])

    hooks = [
        network.model[0].register_forward_hook(capture_preprocessor),
        network.model[spec["detect_index"]].register_forward_pre_hook(
            capture_detect
        ),
    ]
    image = torch.rand(1, 3, imgsz, imgsz)
    try:
        with torch.inference_mode():
            output = network(image)
    finally:
        for hook in hooks:
            hook.remove()
    tensors = list(_iter_tensors(output))
    checks = {
        "preprocessor_shape_preserved": preprocessor_shapes
        == [{"input": list(image.shape), "output": list(image.shape)}],
        "three_detect_inputs": len(detect_inputs) == 3,
        "detect_sizes": [item[-2:] for item in detect_inputs]
        == [[imgsz // 4, imgsz // 4], [imgsz // 8, imgsz // 8], [imgsz // 16, imgsz // 16]],
        "outputs_exist": bool(tensors),
        "outputs_finite": bool(tensors)
        and all(torch.isfinite(item).all().item() for item in tensors),
    }
    return {
        "input_shape": list(image.shape),
        "preprocessor_shapes": preprocessor_shapes,
        "detect_input_shapes": detect_inputs,
        "output_shapes": [list(item.shape) for item in tensors],
        "checks": checks,
        "passed": all(checks.values()),
    }


def backward_report(
    experiment: str,
    *,
    imgsz: int = 64,
) -> dict[str, Any]:
    model = build_model(experiment)
    network = model.model.cpu().train()
    image = torch.rand(1, 3, imgsz, imgsz, requires_grad=True)
    output = network(image)
    tensors = list(_iter_tensors(output))
    loss = sum(item.float().square().mean() for item in tensors)
    loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in network.named_parameters()
        if parameter.grad is not None
    }
    preprocessor_gradients = {
        name: value
        for name, value in gradients.items()
        if name.startswith("model.0.")
    }
    detector_gradients = {
        name: value
        for name, value in gradients.items()
        if not name.startswith("model.0.")
    }
    spec = EXPERIMENTS[experiment]
    branch_checks = {
        "filter_parameter_head": any(
            marker in name
            for name in preprocessor_gradients
            for marker in (
                "encoder.projection",
                "encoder.filter_head",
            )
        ),
        "detector": bool(detector_gradients),
    }
    if spec["preprocessor"] == "vgup":
        branch_checks.update(
            {
                "global_gate_head": any(
                    "global_gate_head" in name
                    for name in preprocessor_gradients
                ),
                "spatial_gate_head": any(
                    "spatial_gate_head" in name
                    for name in preprocessor_gradients
                ),
            }
        )
    finite = all(
        torch.isfinite(value).all().item()
        for value in gradients.values()
    )
    return {
        "input_shape": list(image.shape),
        "loss": float(loss.detach()),
        "preprocessor_gradient_tensors": len(preprocessor_gradients),
        "detector_gradient_tensors": len(detector_gradients),
        "branch_checks": branch_checks,
        "finite": finite,
        "passed": bool(
            image.grad is not None
            and torch.isfinite(image.grad).all().item()
            and preprocessor_gradients
            and detector_gradients
            and finite
            and all(branch_checks.values())
        ),
    }


def state_dict_roundtrip_report(
    experiment: str,
    *,
    imgsz: int = 64,
) -> dict[str, Any]:
    source = build_model(experiment)
    source_state = {
        key: value.detach().cpu().clone()
        for key, value in source.model.state_dict().items()
    }
    with tempfile.TemporaryDirectory(prefix="ship-yolo-erup-vgup-") as directory:
        path = Path(directory) / "state_dict.pt"
        torch.save(source_state, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    target = build_model(experiment)
    result = target.model.load_state_dict(loaded, strict=True)
    forward = forward_report(target, experiment, imgsz=imgsz)
    return {
        "state_tensors": len(source_state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "reload_forward_passed": forward["passed"],
        "passed": (
            not result.missing_keys
            and not result.unexpected_keys
            and forward["passed"]
        ),
    }


def compatibility_report() -> dict[str, Any]:
    register_modules()
    from ultralytics import YOLO
    from tools.cumulative_models_utils import build_model as build_base_model

    native = YOLO("yolo11n.yaml", verbose=False)
    existing_inception = YOLO(
        str(ROOT / "experiments/yolo11n_inceptiondw_c3k2_p23.yaml"),
        verbose=False,
    )
    pls = build_base_model("incdw_dysample_pls")
    pls_scam = build_base_model("incdw_dysample_pls_scam")
    strides = {
        "native": [float(item) for item in native.model.stride],
        "inceptiondw": [
            float(item) for item in existing_inception.model.stride
        ],
        "incdw_dysample_pls": [
            float(item) for item in pls.model.stride
        ],
        "incdw_dysample_pls_scam": [
            float(item) for item in pls_scam.model.stride
        ],
    }
    return {
        "strides": strides,
        "passed": (
            strides["native"] == [8.0, 16.0, 32.0]
            and strides["inceptiondw"] == [8.0, 16.0, 32.0]
            and strides["incdw_dysample_pls"] == EXPECTED_STRIDES
            and strides["incdw_dysample_pls_scam"] == EXPECTED_STRIDES
        ),
    }
