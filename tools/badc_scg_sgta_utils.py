"""Build, audit, and initialize BADC/SCG/SGTA ship-detection experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
BASELINE_YAML = ROOT / "experiments" / "yolo11n_inceptiondw_c3k2_p23.yaml"
VARIANTS: dict[str, dict[str, Any]] = {
    "badc": {
        "yaml": ROOT / "experiments" / "yolo11n_badc_p23.yaml",
        "name": "yolo11n_badc_p23_640",
        "uses_sgta": False,
        "screening": True,
    },
    "scg": {
        "yaml": ROOT / "experiments" / "yolo11n_inceptiondw_scg_p3.yaml",
        "name": "yolo11n_inceptiondw_scg_p3_640",
        "uses_sgta": False,
        "screening": True,
    },
    "sgta": {
        "yaml": ROOT / "experiments" / "yolo11n_inceptiondw_sgta.yaml",
        "name": "yolo11n_inceptiondw_sgta_640",
        "uses_sgta": True,
        "screening": True,
    },
    "full": {
        "yaml": ROOT / "experiments" / "yolo11n_badc_scg_sgta_full.yaml",
        "name": "yolo11n_badc_scg_sgta_full_640",
        "uses_sgta": True,
        "screening": False,
    },
}


def require_ultralytics_version() -> str:
    import ultralytics

    version = getattr(ultralytics, "__version__", "unknown")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Experiments require ultralytics=={EXPECTED_ULTRALYTICS_VERSION}; found {version}."
        )
    return version


def register_modules() -> None:
    require_ultralytics_version()
    from custom_modules.register import register_custom_modules

    register_custom_modules()


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}.")
    return VARIANTS[variant]


def build_model(variant: str, *, seed: int = 0, verbose: bool = False):
    register_modules()
    from ultralytics import YOLO

    torch.manual_seed(seed)
    return YOLO(str(variant_config(variant)["yaml"]), verbose=verbose)


def model_statistics(yolo_model, imgsz: int = 640) -> dict[str, int | float]:
    from ultralytics.utils.torch_utils import get_flops

    network = yolo_model.model
    return {
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in network.parameters() if parameter.requires_grad
        ),
        "gflops_at_imgsz": float(get_flops(network, imgsz=imgsz)),
        "imgsz": imgsz,
    }


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


def transfer_weights(
    target_yolo,
    variant: str,
    weights: str | Path = "yolo11n.pt",
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Transfer every exact-name, exact-shape tensor from official YOLO11n."""

    source_yolo, source_path = _load_source(weights)
    source_state = source_yolo.model.float().state_dict()
    target = target_yolo.model
    target_state = target.state_dict()
    parameter_keys = set(dict(target.named_parameters()))
    updated = dict(target_state)
    inherited: list[str] = []
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is not None and source_tensor.shape == target_tensor.shape:
            updated[key] = source_tensor.to(dtype=target_tensor.dtype)
            inherited.append(key)
    if apply:
        target.load_state_dict(updated, strict=True)

    inherited_parameters = parameter_keys.intersection(inherited)
    report = {
        "variant": variant,
        "source_weights": str(source_path),
        "source_sha256": sha256_file(source_path),
        "loaded_state_tensors": len(inherited),
        "total_state_tensors": len(target_state),
        "loaded_parameter_tensors": len(inherited_parameters),
        "total_parameter_tensors": len(parameter_keys),
        "loaded_target_parameter_elements": sum(
            target_state[key].numel() for key in inherited_parameters
        ),
        "total_target_parameter_elements": sum(
            parameter.numel() for parameter in target.parameters()
        ),
        "random_initialized_parameter_keys": sorted(parameter_keys - inherited_parameters),
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
    variant: str,
    *,
    weights: str | Path,
    output: str | Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Save a deterministic initialization checkpoint and transfer manifest."""

    model = build_model(variant, seed=seed)
    transfer = transfer_weights(model, variant, weights, apply=True)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return {
        "variant": variant,
        "model_yaml": str(variant_config(variant)["yaml"]),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "ultralytics_version": require_ultralytics_version(),
        "git_commit": git_commit(),
        "weight_transfer": transfer,
    }


def forward_report(variant: str, imgsz: int = 128) -> dict[str, Any]:
    model = build_model(variant)
    network = model.model.cpu().eval()
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.inference_mode():
        output = network(x)
    checks = {
        "finite_output": bool(torch.isfinite(output[0]).all()),
        "three_scale_detect": len(output[1]["feats"]) == 3,
        "stride_8_16_32": list(map(float, network.stride)) == [8.0, 16.0, 32.0],
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "prediction_shape": list(output[0].shape),
        "feature_shapes": [list(item.shape) for item in output[1]["feats"]],
    }


def structure_report(variant: str, model=None) -> dict[str, Any]:
    from custom_modules.badc import C3k2_BADC
    from custom_modules.scg import SemanticConfirmationGate

    model = model or build_model(variant)
    layers = model.model.model
    badc_indices = [i for i, layer in enumerate(layers) if isinstance(layer, C3k2_BADC)]
    scg_indices = [
        i for i, layer in enumerate(layers) if isinstance(layer, SemanticConfirmationGate)
    ]
    expected_badc = [2, 4] if variant in {"badc", "full"} else []
    expected_scg = [15] if variant in {"scg", "full"} else []
    head = layers[-1]
    checks = {
        "badc_scope": badc_indices == expected_badc,
        "scg_scope": scg_indices == expected_scg,
        "detect_three_scale": getattr(head, "nl", None) == 3,
        "detect_stride": list(map(float, model.model.stride)) == [8.0, 16.0, 32.0],
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "badc_indices": badc_indices,
        "scg_indices": scg_indices,
        "detect_sources": list(head.f),
    }


def full_audit(
    variant: str,
    *,
    weights: str | Path = "yolo11n.pt",
    imgsz: int = 640,
) -> dict[str, Any]:
    model = build_model(variant)
    structure = structure_report(variant, model)
    forward = forward_report(variant, imgsz=min(imgsz, 256))
    transfer = transfer_weights(model, variant, weights, apply=True)
    checks = {
        "structure": structure["all_checks_passed"],
        "forward": forward["all_checks_passed"],
        "weight_transfer": transfer["loaded_state_tensors"] > 0,
    }
    return {
        "variant": variant,
        "experiment_name": variant_config(variant)["name"],
        "model_yaml": str(variant_config(variant)["yaml"]),
        "uses_sgta": variant_config(variant)["uses_sgta"],
        "screening": variant_config(variant)["screening"],
        "ultralytics_version": require_ultralytics_version(),
        "git_commit": git_commit(),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "structure": structure,
        "forward": forward,
        "statistics": model_statistics(model, imgsz=imgsz),
        "weight_transfer": transfer,
    }
