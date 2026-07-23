"""Build, transfer, audit, and initialization helpers for two small-ship experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
BASELINE_YAML = ROOT / "experiments" / "yolo11n_inceptiondw_c3k2_p23.yaml"
VARIANTS = {
    "spddown": {
        "yaml": ROOT / "experiments" / "yolo11n_inceptiondw_spddown_p3.yaml",
        "name": "yolo11n_inceptiondw_spddown_p3_640",
    },
    "p2_gaussian_aux": {
        "yaml": ROOT / "experiments" / "yolo11n_inceptiondw_p2_gaussian_aux.yaml",
        "name": "yolo11n_inceptiondw_p2_gaussian_aux_640",
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


def build_baseline(*, seed: int = 0, verbose: bool = False):
    register_modules()
    from ultralytics import YOLO

    torch.manual_seed(seed)
    return YOLO(str(BASELINE_YAML), verbose=verbose)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


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


def model_statistics(yolo_model, imgsz: int = 640) -> dict[str, int | float]:
    from ultralytics.utils.torch_utils import get_flops

    network = yolo_model.model
    parameters = sum(parameter.numel() for parameter in network.parameters())
    trainable = sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "gflops_at_imgsz": float(get_flops(network, imgsz=imgsz)),
        "imgsz": imgsz,
    }


def inflate_stride2_conv_to_spd(source_weight: torch.Tensor) -> torch.Tensor:
    """Map a 3x3 stride-2 kernel exactly into S2D + 3x3 stride-1 weights."""

    if source_weight.ndim != 4 or tuple(source_weight.shape[-2:]) != (3, 3):
        raise ValueError(f"Expected OI33 source weight, got {tuple(source_weight.shape)}.")
    out_channels, in_channels = source_weight.shape[:2]
    target = source_weight.new_zeros((out_channels, 4 * in_channels, 3, 3))
    phase_order = ((0, 0), (1, 0), (0, 1), (1, 1))
    phase_to_index = {phase: index for index, phase in enumerate(phase_order)}
    for source_row in range(3):
        for source_col in range(3):
            delta_row = source_row - 1
            delta_col = source_col - 1
            coarse_row = delta_row // 2
            coarse_col = delta_col // 2
            phase = (delta_row - 2 * coarse_row, delta_col - 2 * coarse_col)
            phase_index = phase_to_index[phase]
            channel_slice = slice(phase_index * in_channels, (phase_index + 1) * in_channels)
            target[:, channel_slice, coarse_row + 1, coarse_col + 1] = source_weight[
                :, :, source_row, source_col
            ]
    return target


def _load_source(weights: str | Path):
    register_modules()
    from ultralytics import YOLO

    requested = Path(weights).expanduser()
    if not requested.is_file() and requested.name != "yolo11n.pt":
        raise FileNotFoundError(f"Pretrained weights not found: {requested.resolve()}")
    source = YOLO(str(requested), verbose=False)
    path = Path(source.ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resolved checkpoint does not exist: {path}")
    return source, path


def transfer_weights(
    target_yolo,
    variant: str,
    weights: str | Path = "yolo11n.pt",
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Load exact tensors and apply an exact functional map for SPDDown."""

    source_yolo, source_path = _load_source(weights)
    source_state = source_yolo.model.float().state_dict()
    target = target_yolo.model
    target_state = target.state_dict()
    target_parameter_keys = set(dict(target.named_parameters()))
    updated = dict(target_state)
    inherited: dict[str, str] = {}

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is not None and source_tensor.shape == target_tensor.shape:
            updated[key] = source_tensor.to(dtype=target_tensor.dtype)
            inherited[key] = key

    semantic_mapping: list[dict[str, Any]] = []
    if variant == "spddown":
        source_conv_key = "model.3.conv.weight"
        target_conv_key = "model.3.cv.conv.weight"
        mapped = inflate_stride2_conv_to_spd(source_state[source_conv_key])
        if mapped.shape != target_state[target_conv_key].shape:
            raise RuntimeError(
                f"SPD mapped shape {mapped.shape} != target {target_state[target_conv_key].shape}."
            )
        updated[target_conv_key] = mapped.to(dtype=target_state[target_conv_key].dtype)
        inherited[target_conv_key] = source_conv_key + " (exact SPD inflation)"
        semantic_mapping.append(
            {
                "source": source_conv_key,
                "target": target_conv_key,
                "method": "exact stride2-conv to S2D+stride1-conv inflation",
                "source_elements": source_state[source_conv_key].numel(),
                "target_elements": target_state[target_conv_key].numel(),
                "target_nonzero_elements": int(torch.count_nonzero(mapped)),
            }
        )
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            source_key = f"model.3.bn.{suffix}"
            target_key = f"model.3.cv.bn.{suffix}"
            updated[target_key] = source_state[source_key].to(dtype=target_state[target_key].dtype)
            inherited[target_key] = source_key

    if apply:
        target.load_state_dict(updated, strict=True)

    inherited_parameter_keys = target_parameter_keys.intersection(inherited)
    random_parameter_keys = sorted(target_parameter_keys - inherited_parameter_keys)
    report = {
        "variant": variant,
        "source_weights": str(source_path),
        "source_sha256": sha256_file(source_path),
        "loaded_state_tensors": len(inherited),
        "total_state_tensors": len(target_state),
        "loaded_parameter_tensors": len(inherited_parameter_keys),
        "total_parameter_tensors": len(target_parameter_keys),
        "loaded_target_parameter_elements": sum(
            target_state[key].numel() for key in inherited_parameter_keys
        ),
        "total_target_parameter_elements": sum(parameter.numel() for parameter in target.parameters()),
        "random_initialized_parameter_keys": random_parameter_keys,
        "semantic_mapping": semantic_mapping,
        "matched_source_to_target": [
            {"source": source_key, "target": target_key}
            for target_key, source_key in sorted(inherited.items())
        ],
        "applied": apply,
    }
    report["loaded_tensor_ratio"] = report["loaded_state_tensors"] / report["total_state_tensors"]
    report["loaded_target_parameter_element_ratio"] = (
        report["loaded_target_parameter_elements"] / report["total_target_parameter_elements"]
    )
    return report


def spd_functional_equivalence(weights: str | Path = "yolo11n.pt") -> dict[str, Any]:
    """Verify the semantic initialization reproduces the official layer exactly."""

    source_yolo, _ = _load_source(weights)
    target_yolo = build_model("spddown")
    transfer_weights(target_yolo, "spddown", weights, apply=True)
    source_layer = source_yolo.model.model[3].cpu().eval()
    target_layer = target_yolo.model.model[3].cpu().eval()
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, source_layer.conv.in_channels, 64, 64, generator=generator)
    with torch.inference_mode():
        source_output = source_layer(x)
        target_output = target_layer(x)
    maximum_error = float((source_output - target_output).abs().max())
    return {
        "input_shape": list(x.shape),
        "output_shape": list(source_output.shape),
        "maximum_absolute_error": maximum_error,
        "allclose_at_1e-5": bool(torch.allclose(source_output, target_output, atol=1e-5, rtol=1e-5)),
        "all_checks_passed": bool(
            torch.allclose(source_output, target_output, atol=1e-5, rtol=1e-5)
        ),
    }


def yaml_scope_report(variant: str) -> dict[str, Any]:
    baseline = read_yaml(BASELINE_YAML)
    custom = read_yaml(variant_config(variant)["yaml"])
    baseline_layers = baseline["backbone"] + baseline["head"]
    custom_layers = custom["backbone"] + custom["head"]
    differences = [
        {"index": index, "baseline": base, "custom": changed}
        for index, (base, changed) in enumerate(zip(baseline_layers, custom_layers))
        if base != changed
    ]
    expected = [3] if variant == "spddown" else [23]
    checks = {
        "same_layer_count": len(baseline_layers) == len(custom_layers),
        "single_expected_difference": [item["index"] for item in differences] == expected,
    }
    return {"checks": checks, "all_checks_passed": all(checks.values()), "differences": differences}


def structure_report(variant: str, yolo_model=None) -> dict[str, Any]:
    from ultralytics.nn.modules import Detect

    from custom_modules.p2_gaussian_aux import P2GaussianAuxDetect
    from custom_modules.spd import SPDDown

    yolo_model = yolo_model or build_model(variant)
    layers = yolo_model.model.model
    head = layers[-1]
    spd_indices = [index for index, layer in enumerate(layers) if isinstance(layer, SPDDown)]
    aux_indices = [
        index for index, layer in enumerate(layers) if isinstance(layer, P2GaussianAuxDetect)
    ]
    checks = {
        "detect_is_three_scale": isinstance(head, Detect) and head.nl == 3,
        "detect_strides_are_8_16_32": list(map(float, yolo_model.model.stride)) == [8.0, 16.0, 32.0],
        "spddown_scope": spd_indices == ([3] if variant == "spddown" else []),
        "aux_scope": aux_indices == ([23] if variant == "p2_gaussian_aux" else []),
        "native_detect_sources_or_aux_contract": (
            list(head.f) == [16, 19, 22]
            if variant == "spddown"
            else list(head.f) == [2, 16, 19, 22]
        ),
        "native_neck_layer_count": len(layers) == 24,
    }
    if variant == "p2_gaussian_aux":
        checks.update(
            {
                "aux_weight_is_0_25": head.aux_weight == 0.25,
                "aux_is_one_channel": head.p2_aux.out_channels == 1,
                "detect_channels_exclude_p2": [module[0].conv.in_channels for module in head.cv2]
                == [64, 128, 256],
            }
        )
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "spddown_indices": spd_indices,
        "aux_head_indices": aux_indices,
        "head_sources": list(head.f),
    }


def forward_report(variant: str, imgsz: int = 640) -> dict[str, Any]:
    model = build_model(variant).model.cpu()
    head = model.model[-1]
    detect_inputs: list[list[int]] = []
    auxiliary_calls: list[list[int]] = []
    head_hook = head.register_forward_pre_hook(
        lambda _module, args: detect_inputs.extend([list(tensor.shape) for tensor in args[0]])
    )
    aux_hook = None
    if variant == "p2_gaussian_aux":
        aux_hook = head.p2_aux.register_forward_hook(
            lambda _module, _args, output: auxiliary_calls.append(list(output.shape))
        )
    x = torch.randn(1, 3, imgsz, imgsz)
    try:
        model.eval()
        with torch.inference_mode():
            eval_output = model(x)
        eval_auxiliary_calls = list(auxiliary_calls)
        auxiliary_calls.clear()
        model.train()
        train_output = model(x)
        train_auxiliary_calls = list(auxiliary_calls)
    finally:
        head_hook.remove()
        if aux_hook is not None:
            aux_hook.remove()
    train_predictions = train_output if isinstance(train_output, dict) else {}
    checks = {
        "eval_output_exists": eval_output is not None,
        "three_native_detect_features": len(train_predictions.get("feats", [])) == 3,
        "eval_does_not_execute_aux": not eval_auxiliary_calls,
        "train_aux_contract": (
            "p2_aux_logits" not in train_predictions
            if variant == "spddown"
            else train_auxiliary_calls == [[1, 1, imgsz // 4, imgsz // 4]]
            and list(train_predictions["p2_aux_logits"].shape) == [1, 1, imgsz // 4, imgsz // 4]
        ),
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "head_input_shapes": detect_inputs,
        "eval_auxiliary_calls": eval_auxiliary_calls,
        "train_auxiliary_calls": train_auxiliary_calls,
    }


def auxiliary_gradient_report(imgsz: int = 128) -> dict[str, Any]:
    from custom_modules.p2_gaussian_aux import P2GaussianAuxLoss
    from ultralytics.utils import IterableSimpleNamespace

    yolo_model = build_model("p2_gaussian_aux")
    model = yolo_model.model.cpu().train()
    if isinstance(model.args, dict):
        model.args = IterableSimpleNamespace(**model.args)
    criterion = P2GaussianAuxLoss(model)
    x = torch.randn(2, 3, imgsz, imgsz)
    batch = {
        "img": x,
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.zeros((2, 1)),
        "bboxes": torch.tensor(
            [[0.35, 0.45, 0.04, 0.05], [0.72, 0.60, 0.08, 0.04]],
            dtype=torch.float32,
        ),
    }
    predictions = model(x)
    loss, items = criterion(predictions, batch)
    model.zero_grad(set_to_none=True)
    loss.sum().backward()
    aux_gradient = model.model[-1].p2_aux.weight.grad
    p2_gradient = model.model[2].cv1.conv.weight.grad
    checks = {
        "four_loss_items": list(items.shape) == [4],
        "finite_positive_total": bool(torch.isfinite(loss).all() and loss.sum() > 0),
        "positive_auxiliary_loss": bool(items[-1] > 0),
        "aux_head_has_finite_gradient": aux_gradient is not None
        and bool(torch.isfinite(aux_gradient).all())
        and bool(aux_gradient.abs().sum() > 0),
        "p2_backbone_receives_aux_gradient": p2_gradient is not None
        and bool(torch.isfinite(p2_gradient).all())
        and bool(p2_gradient.abs().sum() > 0),
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "total_loss": float(loss.detach().sum()),
        "loss_items": [float(value) for value in items],
    }


def full_audit(
    variant: str,
    *,
    weights: str | Path = "yolo11n.pt",
    imgsz: int = 640,
) -> dict[str, Any]:
    model = build_model(variant)
    scope = yaml_scope_report(variant)
    structure = structure_report(variant, model)
    forward = forward_report(variant, imgsz=imgsz)
    transfer = transfer_weights(model, variant, weights, apply=True)
    extra = (
        {"spd_functional_equivalence": spd_functional_equivalence(weights)}
        if variant == "spddown"
        else {"auxiliary_gradient": auxiliary_gradient_report(imgsz=min(imgsz, 128))}
    )
    extra_passed = all(item["all_checks_passed"] for item in extra.values())
    checks = {
        "yaml_scope": scope["all_checks_passed"],
        "structure": structure["all_checks_passed"],
        "forward": forward["all_checks_passed"],
        "transfer_has_loaded_tensors": transfer["loaded_state_tensors"] > 0,
        "variant_specific": extra_passed,
    }
    return {
        "variant": variant,
        "experiment_name": variant_config(variant)["name"],
        "model_yaml": str(variant_config(variant)["yaml"]),
        "ultralytics_version": require_ultralytics_version(),
        "git_commit": git_commit(),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "yaml_scope": scope,
        "structure": structure,
        "forward": forward,
        "statistics": model_statistics(model, imgsz=imgsz),
        "weight_transfer": transfer,
        **extra,
    }


def save_initialized_model(
    variant: str,
    *,
    weights: str | Path,
    output: str | Path,
    seed: int = 0,
) -> dict[str, Any]:
    model = build_model(variant, seed=seed)
    transfer = transfer_weights(model, variant, weights, apply=True)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    reloaded = _reload_model(output)
    original_state = model.model.state_dict()
    reloaded_state = reloaded.model.state_dict()
    same_keys = set(original_state) == set(reloaded_state)
    maximum_reload_error = (
        max(
            float(
                (
                    original_state[key].detach().cpu().float()
                    - reloaded_state[key].detach().cpu().float()
                )
                .abs()
                .max()
            )
            for key in original_state
        )
        if same_keys
        else float("inf")
    )
    reload_verified = same_keys and maximum_reload_error <= 2e-3
    if not reload_verified:
        raise RuntimeError(
            "Reloaded initialization checkpoint differs beyond expected FP16 serialization; "
            f"maximum error={maximum_reload_error}."
        )
    return {
        "variant": variant,
        "model_yaml": str(variant_config(variant)["yaml"]),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "ultralytics_version": require_ultralytics_version(),
        "git_commit": git_commit(),
        "weight_transfer": transfer,
        "reload_verified": reload_verified,
        "maximum_fp16_reload_error": maximum_reload_error,
    }


def _reload_model(path: str | Path):
    register_modules()
    from ultralytics import YOLO

    return YOLO(str(path), verbose=False)
