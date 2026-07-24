"""Build, audit, initialize, and report the SCSharedHead experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
MODEL_YAML = ROOT / "experiments" / "yolo11n_inceptiondw_scshared_head.yaml"
BASELINE_YAML = ROOT / "experiments" / "yolo11n_inceptiondw_c3k2_p23.yaml"
EXPERIMENT_NAME = "yolo11n_inceptiondw_scshared_head_640"
DETECT_INDEX = 23


def require_ultralytics_version() -> str:
    import ultralytics

    version = getattr(ultralytics, "__version__", "unknown")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Experiment requires ultralytics=={EXPECTED_ULTRALYTICS_VERSION}; "
            f"found {version}."
        )
    return version


def register_modules() -> None:
    require_ultralytics_version()
    from custom_modules.register import register_custom_modules

    register_custom_modules()


def build_model(*, seed: int = 0, verbose: bool = False):
    register_modules()
    from ultralytics import YOLO

    torch.manual_seed(seed)
    return YOLO(str(MODEL_YAML), verbose=verbose)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def model_statistics(yolo_model, imgsz: int = 640) -> dict[str, int | float]:
    from ultralytics.utils.torch_utils import get_flops

    network = yolo_model.model
    return {
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in network.parameters()
            if parameter.requires_grad
        ),
        "gflops_at_imgsz": float(get_flops(network, imgsz=imgsz)),
        "imgsz": imgsz,
    }


def _load_source(weights: str | Path):
    register_modules()
    from ultralytics import YOLO

    requested = Path(weights).expanduser()
    if not requested.is_file() and requested.name != "yolo11n.pt":
        raise FileNotFoundError(f"Pretrained weights not found: {requested.resolve()}")
    source = YOLO(str(requested), verbose=False)
    source_path = Path(source.ckpt_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Resolved checkpoint does not exist: {source_path}")
    return source, source_path


def _head_mapping_candidates(level: int) -> tuple[tuple[str, str], ...]:
    prefix = f"model.{DETECT_INDEX}"
    return (
        (f"{prefix}.cv2.{level}.weight", f"{prefix}.cv2.{level}.2.weight"),
        (f"{prefix}.cv2.{level}.bias", f"{prefix}.cv2.{level}.2.bias"),
        (f"{prefix}.cv3.{level}.weight", f"{prefix}.cv3.{level}.2.weight"),
        (f"{prefix}.cv3.{level}.bias", f"{prefix}.cv3.{level}.2.bias"),
    )


def transfer_weights(
    target_yolo,
    weights: str | Path = "yolo11n.pt",
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Transfer exact tensors plus shape-safe native Detect output projections."""

    source_yolo, source_path = _load_source(weights)
    source_state = source_yolo.model.float().state_dict()
    target = target_yolo.model
    target_state = target.state_dict()
    parameter_keys = set(dict(target.named_parameters()))
    updated = dict(target_state)
    inherited: dict[str, str] = {}
    exact_keys: list[str] = []
    mapped_keys: list[str] = []

    for target_key, target_tensor in target_state.items():
        source_tensor = source_state.get(target_key)
        if source_tensor is not None and source_tensor.shape == target_tensor.shape:
            updated[target_key] = source_tensor.to(dtype=target_tensor.dtype)
            inherited[target_key] = target_key
            exact_keys.append(target_key)

    for level in range(3):
        for target_key, source_key in _head_mapping_candidates(level):
            if target_key in inherited:
                continue
            target_tensor = target_state.get(target_key)
            source_tensor = source_state.get(source_key)
            if (
                target_tensor is not None
                and source_tensor is not None
                and source_tensor.shape == target_tensor.shape
            ):
                updated[target_key] = source_tensor.to(dtype=target_tensor.dtype)
                inherited[target_key] = source_key
                mapped_keys.append(target_key)

    if apply:
        target.load_state_dict(updated, strict=True)

    inherited_parameter_keys = parameter_keys.intersection(inherited)
    random_keys = sorted(parameter_keys - inherited_parameter_keys)
    head_prefix = f"model.{DETECT_INDEX}."
    report = {
        "source_weights": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_state_tensors": len(source_state),
        "total_state_tensors": len(target_state),
        "loaded_state_tensors": len(inherited),
        "exact_name_state_tensors": len(exact_keys),
        "mapped_native_detect_output_tensors": len(mapped_keys),
        "loaded_parameter_tensors": len(inherited_parameter_keys),
        "total_parameter_tensors": len(parameter_keys),
        "loaded_target_parameter_elements": sum(
            target_state[key].numel() for key in inherited_parameter_keys
        ),
        "total_target_parameter_elements": sum(
            parameter.numel() for parameter in target.parameters()
        ),
        "mapped_head_keys": {
            key: inherited[key] for key in sorted(mapped_keys)
        },
        "random_initialized_head_parameter_keys": [
            key for key in random_keys if key.startswith(head_prefix)
        ],
        "random_initialized_non_head_parameter_keys": [
            key for key in random_keys if not key.startswith(head_prefix)
        ],
        "applied": apply,
    }
    report["loaded_tensor_ratio"] = (
        report["loaded_state_tensors"] / report["total_state_tensors"]
    )
    report["loaded_target_parameter_element_ratio"] = (
        report["loaded_target_parameter_elements"]
        / report["total_target_parameter_elements"]
    )
    return report


def save_initialized_model(
    *,
    weights: str | Path,
    output: str | Path,
    seed: int = 0,
) -> dict[str, Any]:
    model = build_model(seed=seed)
    transfer = transfer_weights(model, weights, apply=True)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return {
        "experiment_name": EXPERIMENT_NAME,
        "model_yaml": str(MODEL_YAML),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "ultralytics_version": require_ultralytics_version(),
        "git_commit": git_commit(),
        "weight_transfer": transfer,
    }


def structure_report(model=None) -> dict[str, Any]:
    from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
    from custom_modules.scshared_head import SCSharedDetect

    model = model or build_model()
    layers = model.model.model
    inception_indices = [
        index
        for index, layer in enumerate(layers)
        if isinstance(layer, C3k2_InceptionDW)
    ]
    head = layers[-1]
    checks = {
        "inceptiondw_scope": inception_indices == [2, 4],
        "head_type": isinstance(head, SCSharedDetect),
        "three_scale_detect": getattr(head, "nl", None) == 3,
        "detect_stride": list(map(float, model.model.stride)) == [8.0, 16.0, 32.0],
        "shared_channels": getattr(head, "shared_channels", None) == 64,
        "two_shared_blocks": len(getattr(head, "shared_stem", [])) == 2,
        "three_scale_calibrators": len(
            getattr(head, "scale_calibration", [])
        ) == 3,
        "single_class": getattr(head, "nc", None) == 1,
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "inceptiondw_indices": inception_indices,
        "detect_sources": list(head.f),
        "head_class": type(head).__name__,
        "scale_values": [
            float(module.log_scale.detach().exp())
            for module in head.scale_calibration
        ],
    }


def forward_report(imgsz: int = 128) -> dict[str, Any]:
    model = build_model()
    network = model.model.cpu()
    sample = torch.randn(2, 3, imgsz, imgsz)

    network.train()
    training_output = network(sample)
    training_checks = {
        "training_dict": isinstance(training_output, dict),
        "boxes_finite": bool(torch.isfinite(training_output["boxes"]).all()),
        "scores_finite": bool(torch.isfinite(training_output["scores"]).all()),
        "three_training_features": len(training_output["feats"]) == 3,
    }

    loss = training_output["boxes"].mean() + training_output["scores"].mean()
    loss.backward()
    head = network.model[-1]
    shared_gradient = head.shared_stem[0].conv.weight.grad
    training_checks["shared_stem_gradient"] = (
        shared_gradient is not None and bool(torch.isfinite(shared_gradient).all())
    )

    network.zero_grad(set_to_none=True)
    network.eval()
    with torch.inference_mode():
        inference_output = network(sample[:1])
    inference_checks = {
        "inference_finite": bool(torch.isfinite(inference_output[0]).all()),
        "prediction_channels": inference_output[0].shape[1] == 5,
        "three_inference_features": len(inference_output[1]["feats"]) == 3,
    }
    checks = {**training_checks, **inference_checks}
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "training_box_shape": list(training_output["boxes"].shape),
        "training_score_shape": list(training_output["scores"].shape),
        "inference_shape": list(inference_output[0].shape),
        "feature_shapes": [
            list(feature.shape) for feature in inference_output[1]["feats"]
        ],
    }


def full_audit(
    *,
    weights: str | Path = "yolo11n.pt",
    imgsz: int = 640,
) -> dict[str, Any]:
    model = build_model()
    structure = structure_report(model)
    forward = forward_report(imgsz=min(imgsz, 256))
    transfer = transfer_weights(model, weights, apply=True)
    expected_inception_prefixes = (
        "model.2.m.0.cv2.",
        "model.2.m.0.cv2_adapter.",
        "model.4.m.0.cv2.",
        "model.4.m.0.cv2_adapter.",
    )
    non_head_random_is_scoped = all(
        key.startswith(expected_inception_prefixes)
        for key in transfer["random_initialized_non_head_parameter_keys"]
    )
    checks = {
        "structure": structure["all_checks_passed"],
        "forward": forward["all_checks_passed"],
        "weight_transfer": transfer["loaded_state_tensors"] > 0,
        "head_mapping": transfer["mapped_native_detect_output_tensors"] >= 6,
        "non_head_random_scope": non_head_random_is_scoped,
        "parameter_element_inheritance": transfer[
            "loaded_target_parameter_element_ratio"
        ] > 0.95,
    }
    return {
        "experiment_name": EXPERIMENT_NAME,
        "model_yaml": str(MODEL_YAML),
        "baseline_yaml": str(BASELINE_YAML),
        "ultralytics_version": require_ultralytics_version(),
        "git_commit": git_commit(),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "structure": structure,
        "forward": forward,
        "statistics": model_statistics(model, imgsz=imgsz),
        "weight_transfer": transfer,
    }
