"""Construction and audit helpers for the CA-SCAM experiment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
EXPECTED_STRIDES = [4.0, 8.0, 16.0]
EXPERIMENT = "incdw_dysample_pls_ca_scam_vgup"
MODEL_YAML = ROOT / "experiments/yolo11n_incdw_dysample_pls_ca_scam_vgup.yaml"
BASE_EXPERIMENT = "incdw_dysample_sfl_scam_vgup"
CA_LAYER_INDICES = (22, 23, 24)
DETECT_INDEX = 25
DETECT_FROM = [22, 23, 24]


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
    from custom_modules.register import register_calibrated_scam_modules

    register_calibrated_scam_modules()


def build_model(*, nc: int | None = None, verbose: bool = False):
    register_modules()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    wrapper = YOLO(str(MODEL_YAML), verbose=verbose)
    current_nc = int(wrapper.model.model[-1].nc)
    if nc is not None and current_nc != int(nc):
        wrapper.model = DetectionModel(
            cfg=str(MODEL_YAML),
            ch=3,
            nc=int(nc),
            verbose=verbose,
        )
        wrapper.task = "detect"
        wrapper.ckpt = wrapper.ckpt or {}
    return wrapper


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def expected_ca_keys() -> list[str]:
    suffixes = (
        "contrast_logit",
        "contrast_proj.weight",
        "contrast_proj.bias",
    )
    return sorted(
        f"model.{index}.{suffix}"
        for index in CA_LAYER_INDICES
        for suffix in suffixes
    )


def initialize_from_official(
    target,
    *,
    weights: str | Path = "yolo11n.pt",
    apply: bool = True,
) -> dict[str, Any]:
    """Initialize the successful topology from official weights, then copy shared tensors.

    This intentionally does not use the successful experiment's ``best.pt``.
    The base and CA models receive the same official YOLO11n initialization;
    only CA-SCAM's nine new tensors remain at their deterministic defaults.
    """

    from tools.erup_vgup_utils import (
        build_model as build_base_model,
        initialize_from_official as initialize_base,
    )

    target_nc = int(target.model.model[-1].nc)
    torch.manual_seed(0)
    base = build_base_model(BASE_EXPERIMENT, nc=target_nc, verbose=False)
    official = initialize_base(
        base,
        BASE_EXPERIMENT,
        weights=weights,
        apply=True,
    )
    if not official["passed"]:
        raise RuntimeError(
            "Official YOLO11n initialization of the base topology failed: "
            f"{official['verification_failures']}"
        )

    source_state = base.model.state_dict()
    target_state = target.model.state_dict()
    copied: dict[str, torch.Tensor] = {}
    shape_mismatches: list[dict[str, Any]] = []
    for key, value in target_state.items():
        source = source_state.get(key)
        if source is None:
            continue
        if tuple(source.shape) != tuple(value.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(source.shape),
                    "target_shape": list(value.shape),
                }
            )
            continue
        copied[key] = source.detach().cpu()

    missing = sorted(set(target_state) - set(copied))
    expected_missing = expected_ca_keys()
    missing_after_load: list[str] = []
    unexpected_after_load: list[str] = []
    verification_failures: list[str] = []
    if apply:
        result = target.model.load_state_dict(copied, strict=False)
        missing_after_load = sorted(result.missing_keys)
        unexpected_after_load = sorted(result.unexpected_keys)
        loaded = target.model.state_dict()
        verification_failures = [
            key
            for key, expected in copied.items()
            if not torch.equal(loaded[key].detach().cpu(), expected)
        ]

    new_state = target.model.state_dict()
    zero_initialization = {
        key: bool(torch.count_nonzero(new_state[key]).item() == 0)
        for key in expected_missing
    }
    scam_shared_keys = sorted(
        key
        for key in copied
        if key.startswith(tuple(f"model.{i}." for i in CA_LAYER_INDICES))
    )
    passed = (
        not shape_mismatches
        and missing == expected_missing
        and (not apply or missing_after_load == expected_missing)
        and not unexpected_after_load
        and not verification_failures
        and all(zero_initialization.values())
    )
    return {
        "experiment": EXPERIMENT,
        "base_experiment": BASE_EXPERIMENT,
        "initialization_source": (
            "official YOLO11n weights through the successful base topology; "
            "not the successful experiment best.pt"
        ),
        "official_weights": str(weights),
        "official_to_base": official,
        "loaded_tensors": len(copied),
        "total_tensors": len(target_state),
        "loaded_total": f"{len(copied)}/{len(target_state)}",
        "missing_new_ca_tensors": missing,
        "expected_new_ca_tensors": expected_missing,
        "shape_mismatches": shape_mismatches,
        "missing_after_load": missing_after_load,
        "unexpected_after_load": unexpected_after_load,
        "verification_failures": verification_failures,
        "shared_scam_tensors_loaded": len(scam_shared_keys),
        "shared_scam_keys": scam_shared_keys,
        "new_ca_zero_initialization": zero_initialization,
        "applied": apply,
        "passed": passed,
    }


def model_statistics(model, *, imgsz: int = 640) -> dict[str, Any]:
    from ultralytics.utils.torch_utils import get_flops

    network = model.model
    return {
        "parameters": sum(p.numel() for p in network.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in network.parameters() if p.requires_grad
        ),
        "state_tensors": len(network.state_dict()),
        "gflops": float(get_flops(network, imgsz=imgsz)),
    }


def structure_report(model) -> dict[str, Any]:
    from custom_modules.calibrated_scam import CASCAM
    from custom_modules.dysample import DySample
    from custom_modules.scam import SCAM
    from custom_modules.vgup import VGUPPreprocessor
    from tools.erup_vgup_utils import build_model as build_base_model

    network = model.model
    layers = network.model
    base_layers = build_base_model(
        BASE_EXPERIMENT,
        nc=int(layers[-1].nc),
    ).model.model
    ca_layers = [index for index, layer in enumerate(layers) if type(layer) is CASCAM]
    unchanged = all(
        (
            type(layer) is type(base_layers[index])
            if index not in CA_LAYER_INDICES
            else type(base_layers[index]) is SCAM and type(layer) is CASCAM
        )
        for index, layer in enumerate(layers)
    )
    ca_parameter_ids = [
        {id(parameter) for parameter in layers[index].parameters()}
        for index in CA_LAYER_INDICES
    ]
    independent = all(
        ca_parameter_ids[left].isdisjoint(ca_parameter_ids[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    detector = layers[DETECT_INDEX]
    checks = {
        "only_scam_types_changed": unchanged,
        "vgup_is_first": isinstance(layers[0], VGUPPreprocessor),
        "two_dysample_blocks": sum(isinstance(layer, DySample) for layer in layers) == 2,
        "three_ca_scam_blocks": ca_layers == list(CA_LAYER_INDICES),
        "no_plain_scam_blocks": not any(type(layer) is SCAM for layer in layers),
        "ca_instances_independent": independent,
        "detect_from": list(detector.f) == DETECT_FROM,
        "strides": [float(item) for item in network.stride] == EXPECTED_STRIDES,
    }
    return {
        "layer_types": [type(layer).__name__ for layer in layers],
        "ca_scam_indices": ca_layers,
        "detect_index": DETECT_INDEX,
        "detect_from": list(detector.f),
        "strides": [float(item) for item in network.stride],
        "checks": checks,
        "passed": all(checks.values()),
    }


def forward_report(model, *, imgsz: int = 640) -> dict[str, Any]:
    network = model.model.cpu().eval()
    ca_inputs: list[list[int]] = []
    detect_inputs: list[list[int]] = []
    hooks = []

    for index in CA_LAYER_INDICES:
        hooks.append(
            network.model[index].register_forward_pre_hook(
                lambda _module, args, store=ca_inputs: store.append(
                    list(args[0].shape)
                )
            )
        )
    hooks.append(
        network.model[DETECT_INDEX].register_forward_pre_hook(
            lambda _module, args: detect_inputs.extend(
                list(item.shape) for item in args[0]
            )
        )
    )
    image = torch.rand(1, 3, imgsz, imgsz)
    try:
        with torch.inference_mode():
            output = network(image)
    finally:
        for hook in hooks:
            hook.remove()

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)

    outputs = list(tensors(output))
    expected_sizes = [
        [imgsz // 4, imgsz // 4],
        [imgsz // 8, imgsz // 8],
        [imgsz // 16, imgsz // 16],
    ]
    checks = {
        "three_ca_inputs": len(ca_inputs) == 3,
        "ca_input_sizes": [shape[-2:] for shape in ca_inputs] == expected_sizes,
        "detect_input_sizes": [shape[-2:] for shape in detect_inputs] == expected_sizes,
        "outputs_exist": bool(outputs),
        "outputs_finite": bool(outputs)
        and all(torch.isfinite(item).all().item() for item in outputs),
    }
    return {
        "input_shape": list(image.shape),
        "ca_input_shapes": ca_inputs,
        "detect_input_shapes": detect_inputs,
        "output_shapes": [list(item.shape) for item in outputs],
        "checks": checks,
        "passed": all(checks.values()),
    }


def backward_report(*, imgsz: int = 64) -> dict[str, Any]:
    model = build_model()
    network = model.model.cpu().train()
    image = torch.rand(1, 3, imgsz, imgsz, requires_grad=True)
    output = network(image)

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)

    loss = sum(item.float().square().mean() for item in tensors(output))
    loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in network.named_parameters()
        if parameter.grad is not None
    }
    logit_gradients = [
        name for name in gradients if name.endswith("contrast_logit")
    ]
    shared_scam_gradients = [
        name
        for name in gradients
        if any(name.startswith(f"model.{i}.k.") for i in CA_LAYER_INDICES)
    ]
    detector_gradients = [
        name for name in gradients if name.startswith(f"model.{DETECT_INDEX}.")
    ]
    finite = all(torch.isfinite(value).all().item() for value in gradients.values())
    return {
        "loss": float(loss.detach()),
        "gradient_tensors": len(gradients),
        "contrast_logit_gradients": logit_gradients,
        "shared_scam_gradients": shared_scam_gradients,
        "detector_gradients": detector_gradients,
        "finite": finite,
        "passed": bool(
            image.grad is not None
            and len(logit_gradients) == 3
            and shared_scam_gradients
            and detector_gradients
            and finite
        ),
    }


def state_dict_roundtrip_report(*, imgsz: int = 64) -> dict[str, Any]:
    source = build_model()
    state = {
        key: value.detach().cpu().clone()
        for key, value in source.model.state_dict().items()
    }
    with tempfile.TemporaryDirectory(prefix="ship-yolo-ca-scam-") as directory:
        path = Path(directory) / "state_dict.pt"
        torch.save(state, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    target = build_model()
    result = target.model.load_state_dict(loaded, strict=True)
    forward = forward_report(target, imgsz=imgsz)
    return {
        "state_tensors": len(state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "reload_forward_passed": forward["passed"],
        "passed": (
            not result.missing_keys
            and not result.unexpected_keys
            and forward["passed"]
        ),
    }
