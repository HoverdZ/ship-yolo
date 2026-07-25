"""Shared construction and audit helpers for cumulative YOLO11n experiments."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "incdw_dysample": {
        "yaml": ROOT / "experiments/yolo11n_inceptiondw_dysample.yaml",
        "strides": [8.0, 16.0, 32.0],
        "detect_from": [16, 19, 22],
        "detect_index": 23,
        "uses_pls": False,
        "uses_scam": False,
    },
    "incdw_dysample_pls": {
        "yaml": ROOT
        / "experiments/yolo11n_inceptiondw_dysample_pls.yaml",
        "strides": [4.0, 8.0, 16.0],
        "detect_from": [14, 17, 20],
        "detect_index": 21,
        "uses_pls": True,
        "uses_scam": False,
    },
    "incdw_dysample_pls_scam": {
        "yaml": ROOT
        / "experiments/yolo11n_inceptiondw_dysample_pls_scam.yaml",
        "strides": [4.0, 8.0, 16.0],
        "detect_from": [21, 22, 23],
        "detect_index": 24,
        "uses_pls": True,
        "uses_scam": True,
    },
}


def require_ultralytics_version() -> str:
    import ultralytics

    version = getattr(ultralytics, "__version__", "unknown")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected ultralytics=={EXPECTED_ULTRALYTICS_VERSION}, "
            f"found {version}."
        )
    return version


def register_modules() -> None:
    require_ultralytics_version()
    from custom_modules.register import register_cumulative_modules

    register_cumulative_modules()


def build_model(
    experiment: str,
    *,
    nc: int | None = None,
    verbose: bool = False,
):
    if experiment not in EXPERIMENTS:
        raise KeyError(f"Unknown cumulative experiment: {experiment}")
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


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def read_dataset_nc(path: str | Path) -> int:
    data_yaml = Path(path)
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. "
            "Refusing to fall back to a built-in dataset."
        )
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    nc = payload.get("nc")
    if not isinstance(nc, int) or nc <= 0:
        raise ValueError(f"Invalid dataset nc in {data_yaml}: {nc!r}")
    for split in ("train", "val", "test"):
        if split not in payload:
            raise KeyError(f"Dataset YAML is missing {split!r}: {data_yaml}")
    return nc


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def model_statistics(model, *, imgsz: int = 640) -> dict[str, Any]:
    from ultralytics.utils.torch_utils import get_flops

    network = model.model
    return {
        "parameters": sum(item.numel() for item in network.parameters()),
        "trainable_parameters": sum(
            item.numel()
            for item in network.parameters()
            if item.requires_grad
        ),
        "state_tensors": len(network.state_dict()),
        "gflops": float(get_flops(network, imgsz=imgsz)),
    }


def structure_report(model, experiment: str) -> dict[str, Any]:
    from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
    from custom_modules.dysample import DySample
    from custom_modules.scam import SCAM
    from ultralytics.nn.modules import C2PSA, SPPF

    spec = EXPERIMENTS[experiment]
    layers = model.model.model
    detect = layers[spec["detect_index"]]
    inception_indices = [
        index
        for index, layer in enumerate(layers)
        if isinstance(layer, C3k2_InceptionDW)
    ]
    dysample_indices = [
        index
        for index, layer in enumerate(layers)
        if isinstance(layer, DySample)
    ]
    scam_indices = [
        index
        for index, layer in enumerate(layers)
        if isinstance(layer, SCAM)
    ]
    scam_parameter_ids = [
        {id(parameter) for parameter in layers[index].parameters()}
        for index in scam_indices
    ]
    scam_disjoint = all(
        left.isdisjoint(right)
        for offset, left in enumerate(scam_parameter_ids)
        for right in scam_parameter_ids[offset + 1 :]
    )

    pls_backbone_passed = True
    if spec["uses_pls"]:
        backbone = model.model.yaml["backbone"]
        pls_backbone_passed = (
            len(backbone) == 9
            and isinstance(layers[7], SPPF)
            and isinstance(layers[8], C2PSA)
        )

    checks = {
        "inceptiondw_only_p2_p3": inception_indices == [2, 4],
        "exactly_two_dysample": len(dysample_indices) == 2,
        "detect_from": list(detect.f) == spec["detect_from"],
        "detect_strides": [
            float(value) for value in model.model.stride
        ]
        == spec["strides"],
        "pls_backbone_has_no_p5": pls_backbone_passed,
        "scam_count": len(scam_indices)
        == (3 if spec["uses_scam"] else 0),
        "scam_parameters_independent": scam_disjoint,
    }
    return {
        "inceptiondw_indices": inception_indices,
        "dysample_indices": dysample_indices,
        "scam_indices": scam_indices,
        "detect_index": spec["detect_index"],
        "detect_from": list(detect.f),
        "strides": [float(value) for value in model.model.stride],
        "checks": checks,
        "passed": all(checks.values()),
    }


def forward_report(
    model,
    experiment: str,
    *,
    imgsz: int,
) -> dict[str, Any]:
    from custom_modules.dysample import DySample
    from custom_modules.scam import SCAM

    spec = EXPERIMENTS[experiment]
    network = model.model.cpu().eval()
    detect_inputs: list[list[int]] = []
    dysample_shapes: list[dict[str, list[int]]] = []
    scam_shapes: list[dict[str, list[int]]] = []
    hooks = []

    def capture_detect(_module, args) -> None:
        detect_inputs.extend(list(item.shape) for item in args[0])

    def capture_dysample(_module, args, output) -> None:
        dysample_shapes.append(
            {
                "input": list(args[0].shape),
                "output": list(output.shape),
            }
        )

    def capture_scam(_module, args, output) -> None:
        scam_shapes.append(
            {
                "input": list(args[0].shape),
                "output": list(output.shape),
            }
        )

    hooks.append(
        network.model[spec["detect_index"]].register_forward_pre_hook(
            capture_detect
        )
    )
    for module in network.modules():
        if isinstance(module, DySample):
            hooks.append(module.register_forward_hook(capture_dysample))
        elif isinstance(module, SCAM):
            hooks.append(module.register_forward_hook(capture_scam))

    generator = torch.Generator(device="cpu").manual_seed(0)
    image = torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        generator=generator,
    )
    try:
        with torch.inference_mode():
            output = network(image)
    finally:
        for hook in hooks:
            hook.remove()

    tensors = list(_iter_tensors(output))
    expected_sizes = [
        [imgsz // int(stride), imgsz // int(stride)]
        for stride in spec["strides"]
    ]
    checks = {
        "outputs_exist": bool(tensors),
        "outputs_finite": bool(tensors)
        and all(torch.isfinite(item).all().item() for item in tensors),
        "three_detect_inputs": len(detect_inputs) == 3,
        "detect_sizes": [shape[-2:] for shape in detect_inputs]
        == expected_sizes,
        "dysample_channels_preserved": all(
            item["input"][1] == item["output"][1]
            for item in dysample_shapes
        ),
        "dysample_doubles_spatial_size": all(
            item["output"][-2] == 2 * item["input"][-2]
            and item["output"][-1] == 2 * item["input"][-1]
            for item in dysample_shapes
        ),
        "scam_shapes_preserved": all(
            item["input"] == item["output"] for item in scam_shapes
        ),
    }
    return {
        "input_shape": list(image.shape),
        "detect_input_shapes": detect_inputs,
        "dysample_shapes": dysample_shapes,
        "scam_shapes": scam_shapes,
        "output_tensor_shapes": [list(item.shape) for item in tensors],
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
    generator = torch.Generator(device="cpu").manual_seed(0)
    image = torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        generator=generator,
        requires_grad=True,
    )
    output = network(image)
    tensors = list(_iter_tensors(output))
    if not tensors:
        raise RuntimeError("Backward smoke forward returned no tensors.")
    loss = sum(item.float().square().mean() for item in tensors)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in network.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    passed = bool(
        image.grad is not None
        and torch.isfinite(image.grad).all().item()
        and gradients
        and all(torch.isfinite(item).all().item() for item in gradients)
    )
    return {
        "input_shape": list(image.shape),
        "loss": float(loss.detach()),
        "parameter_gradients": len(gradients),
        "passed": passed,
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
    with tempfile.TemporaryDirectory(prefix="ship-yolo-state-") as directory:
        checkpoint = Path(directory) / "state_dict.pt"
        torch.save(source_state, checkpoint)
        loaded = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
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


def transfer_pretrained_weights(
    target_model,
    weights: str | Path = "yolo11n.pt",
    *,
    apply: bool,
) -> dict[str, Any]:
    register_modules()
    from ultralytics import YOLO

    source = YOLO(str(weights), verbose=False)
    source_state = source.model.float().state_dict()
    target_state = target_model.model.state_dict()
    target_parameter_names = set(dict(target_model.model.named_parameters()))
    compatible = {
        key: value.detach().cpu()
        for key, value in source_state.items()
        if key in target_state
        and tuple(value.shape) == tuple(target_state[key].shape)
    }
    compatible_keys = set(compatible)
    unmatched_target = sorted(set(target_state) - compatible_keys)
    unmatched_source = sorted(set(source_state) - compatible_keys)
    parameter_elements = sum(
        item.numel() for item in target_model.model.parameters()
    )
    inherited_parameter_elements = sum(
        target_state[key].numel()
        for key in compatible_keys
        if key in target_parameter_names
    )

    load_missing: list[str] = []
    load_unexpected: list[str] = []
    verification_failures: list[str] = []
    if apply:
        result = target_model.model.load_state_dict(
            compatible,
            strict=False,
        )
        load_missing = list(result.missing_keys)
        load_unexpected = list(result.unexpected_keys)
        loaded_state = target_model.model.state_dict()
        verification_failures = [
            key
            for key, expected in compatible.items()
            if not torch.equal(
                loaded_state[key].detach().cpu(),
                expected,
            )
        ]

    def module_prefix(key: str) -> str:
        parts = key.split(".")
        return ".".join(parts[: min(4, len(parts))])

    return {
        "weights": str(weights),
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
        "inherited_tensors": len(compatible),
        "unmatched_target_tensors": len(unmatched_target),
        "tensor_inheritance_ratio": len(compatible) / len(target_state),
        "target_parameter_elements": parameter_elements,
        "inherited_parameter_elements": inherited_parameter_elements,
        "parameter_element_inheritance_ratio": (
            inherited_parameter_elements / parameter_elements
        ),
        "unmatched_target_keys": unmatched_target,
        "unmatched_source_keys": unmatched_source,
        "major_unmatched_modules": sorted(
            {module_prefix(key) for key in unmatched_target}
        ),
        "load_missing_keys": load_missing,
        "load_missing_keys_are_reported_unmatched": set(
            load_missing
        ).issubset(set(unmatched_target)),
        "load_unexpected_keys": load_unexpected,
        "verification_failures": verification_failures,
        "applied": apply,
        "passed": (
            not apply
            or (
                set(load_missing).issubset(set(unmatched_target))
                and not load_unexpected
                and not verification_failures
            )
        ),
    }


def compatibility_report() -> dict[str, Any]:
    register_modules()
    from ultralytics import YOLO

    native = YOLO("yolo11n.yaml", verbose=False)
    existing = YOLO(
        str(ROOT / "experiments/yolo11n_inceptiondw_c3k2_p23.yaml"),
        verbose=False,
    )
    return {
        "native_yolo11n_strides": [
            float(value) for value in native.model.stride
        ],
        "existing_inceptiondw_strides": [
            float(value) for value in existing.model.stride
        ],
        "passed": (
            [float(value) for value in native.model.stride]
            == [8.0, 16.0, 32.0]
            and [float(value) for value in existing.model.stride]
            == [8.0, 16.0, 32.0]
        ),
    }
