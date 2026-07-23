"""Shared ASCGD build, audit, profiling, and initialization helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ULTRALYTICS_VERSION = "8.4.92"
CONFIG_PATH = ROOT / "configs" / "ascgd_experiments.yaml"
BASELINE_YAML = ROOT / "experiments" / "yolo11n_inceptiondw_c3k2_p23.yaml"
REPORT_DIR = ROOT / "reports" / "ascgd_preflight"
DEFAULT_WEIGHTS = ROOT / "yolo11n.pt"
BACKBONE_LAST_INDEX = 10
IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

VARIANTS: dict[str, dict[str, str]] = {
    "a_base": {
        "yaml": "experiments/yolo11n_incdw_ascgd_a_base.yaml",
        "name": "yolo11n_incdw_ascgd_base_640",
    },
    "b_gather": {
        "yaml": "experiments/yolo11n_incdw_ascgd_b_gather.yaml",
        "name": "yolo11n_incdw_ascgd_gather_640",
    },
    "c_sca": {
        "yaml": "experiments/yolo11n_incdw_ascgd_c_sca.yaml",
        "name": "yolo11n_incdw_ascgd_sca_640",
    },
    "d_cca": {
        "yaml": "experiments/yolo11n_incdw_ascgd_d_cca.yaml",
        "name": "yolo11n_incdw_ascgd_cca_640",
    },
    "e_full": {
        "yaml": "experiments/yolo11n_incdw_ascgd_e_full.yaml",
        "name": "yolo11n_incdw_ascgd_full_640",
    },
    "f_swap": {
        "yaml": "experiments/yolo11n_incdw_ascgd_f_swap.yaml",
        "name": "yolo11n_incdw_ascgd_swap_640",
    },
    "g_symmetric": {
        "yaml": "experiments/yolo11n_incdw_ascgd_g_symmetric.yaml",
        "name": "yolo11n_incdw_ascgd_symmetric_640",
    },
}


def variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}.")
    value = dict(VARIANTS[variant])
    value["variant"] = variant
    value["yaml_path"] = ROOT / value["yaml"]
    return value


def require_ultralytics_version() -> str:
    import ultralytics

    version = getattr(ultralytics, "__version__", "unknown")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"ASCGD requires ultralytics=={EXPECTED_ULTRALYTICS_VERSION}; found {version}."
        )
    return version


def register_modules() -> None:
    require_ultralytics_version()
    from custom_modules.register import register_ascgd_modules

    register_ascgd_modules()


def build_model(variant: str, *, seed: int = 0, verbose: bool = False):
    """Build a deterministic variant without modifying installed Ultralytics."""

    register_modules()
    from ultralytics import YOLO

    config = variant_config(variant)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return YOLO(str(config["yaml_path"]), verbose=verbose)


def build_baseline(*, nc: int = 1, seed: int = 0, verbose: bool = False):
    """Build the validated InceptionDW baseline with its formal dataset nc override."""

    register_modules()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        baseline = YOLO(str(BASELINE_YAML), verbose=verbose)
        if int(baseline.model.yaml["nc"]) != nc:
            config = deepcopy(baseline.model.yaml)
            config["nc"] = nc
            baseline.model = DetectionModel(
                config,
                ch=3,
                nc=nc,
                verbose=verbose,
            )
        return baseline


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: str | Path) -> str:
    """Use a repository-relative path when possible, otherwise an absolute path."""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _shape_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_tree(item) for item in value]
    return type(value).__name__


def iter_tensors(value: Any) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def detect_index(network: nn.Module) -> int:
    from ultralytics.nn.modules import Detect

    indices = [
        index
        for index, layer in enumerate(network.model)
        if isinstance(layer, Detect)
    ]
    if len(indices) != 1:
        raise RuntimeError(f"Expected one Detect layer, found indices {indices}.")
    return indices[0]


def layer_signature(network: nn.Module, *, stop: int | None = None) -> list[dict[str, Any]]:
    layers = network.model if stop is None else network.model[:stop]
    return [
        {
            "index": index,
            "type": layer.__class__.__module__ + "." + layer.__class__.__qualname__,
            "from": deepcopy(layer.f),
            "parameter_shapes": {
                name: list(value.shape)
                for name, value in layer.state_dict().items()
            },
        }
        for index, layer in enumerate(layers)
    ]


def state_shape_signature(network: nn.Module, prefix: str | None = None) -> dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in network.state_dict().items()
        if prefix is None or key.startswith(prefix)
    }


def module_inventory(network: nn.Module) -> dict[str, Any]:
    counts = Counter(module.__class__.__name__ for module in network.modules())
    parameters: dict[str, int] = {}
    for name, module in network.named_modules():
        own = sum(parameter.numel() for parameter in module.parameters(recurse=False))
        if own:
            parameters[name or "<root>"] = own
    return {
        "module_type_counts": dict(sorted(counts.items())),
        "parameterized_modules": parameters,
    }


def model_statistics(yolo_model, imgsz: int = 640) -> dict[str, Any]:
    network = yolo_model.model
    parameters = sum(parameter.numel() for parameter in network.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    result: dict[str, Any] = {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "fp32_parameter_size_mib": parameters * 4 / (1024**2),
        "gflops": None,
        "gflops_error": None,
        "gflops_method": "Ultralytics get_flops/THOP; attention matmul cost may be uncounted",
    }
    try:
        from ultralytics.utils.torch_utils import get_flops

        result["gflops"] = float(get_flops(network, imgsz=imgsz))
    except Exception as exc:  # report the concrete profiler failure; never hide it
        result["gflops_error"] = f"{type(exc).__name__}: {exc}"
    return result


def forward_signature(
    yolo_model,
    *,
    imgsz: int | tuple[int, int] = 640,
    batch: int = 2,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run one forward and capture the exact three tensors passed to Detect."""

    network = yolo_model.model.to(device).eval()
    index = detect_index(network)
    detect_inputs: list[list[int]] = []

    def capture(_module: nn.Module, args: tuple[Any, ...]) -> None:
        values = args[0]
        detect_inputs.extend([list(value.shape) for value in values])

    hook = network.model[index].register_forward_pre_hook(capture)
    generator = torch.Generator(device="cpu").manual_seed(17)
    height, width = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid input size {(height, width)}.")
    x = torch.randn(batch, 3, height, width, generator=generator).to(device)
    try:
        with torch.inference_mode():
            output = network(x)
    finally:
        hook.remove()
    tensors = list(iter_tensors(output))
    return {
        "input_shape": list(x.shape),
        "detect_index": index,
        "detect_input_shapes": detect_inputs,
        "detect_spatial_sizes": [shape[-2:] for shape in detect_inputs],
        "detect_strides": [float(value) for value in network.stride.tolist()],
        "output_tree": _shape_tree(output),
        "tensor_count": len(tensors),
        "all_finite": bool(tensors)
        and all(bool(torch.isfinite(value).all()) for value in tensors),
    }


def baseline_equivalence() -> dict[str, Any]:
    """Compare A with the effective one-class formal InceptionDW model."""

    baseline = build_baseline(nc=1, seed=23)
    candidate = build_model("a_base", seed=23)
    baseline_stats = model_statistics(baseline)
    candidate_stats = model_statistics(candidate)
    baseline_shapes = state_shape_signature(baseline.model)
    candidate_shapes = state_shape_signature(candidate.model)
    checks = {
        "layer_structure": layer_signature(baseline.model)
        == layer_signature(candidate.model),
        "parameter_count": baseline_stats["parameters"]
        == candidate_stats["parameters"],
        "trainable_parameter_count": baseline_stats["trainable_parameters"]
        == candidate_stats["trainable_parameters"],
        "gflops": baseline_stats["gflops"] is not None
        and candidate_stats["gflops"] is not None
        and abs(baseline_stats["gflops"] - candidate_stats["gflops"]) < 1.0e-12,
        "state_dict_keys": list(baseline_shapes) == list(candidate_shapes),
        "state_dict_shapes": baseline_shapes == candidate_shapes,
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "class_count": 1,
        "baseline_nc_provenance": (
            "The historical YAML stores nc=80, while formal ship data overrides it "
            "to nc=1. This comparison builds that effective one-class baseline."
        ),
        "baseline_statistics": baseline_stats,
        "candidate_statistics": candidate_stats,
    }


def backbone_fairness(models: dict[str, Any]) -> dict[str, Any]:
    baseline = models["a_base"].model
    reference_layers = layer_signature(baseline, stop=BACKBONE_LAST_INDEX + 1)
    reference_state = state_shape_signature(baseline, prefix="model.")
    reference_backbone_state = {
        key: value
        for key, value in reference_state.items()
        if int(key.split(".")[1]) <= BACKBONE_LAST_INDEX
    }
    reference_detect_index = detect_index(baseline)
    reference_detect = baseline.model[reference_detect_index]
    reference_detect_state = {
        key: list(value.shape)
        for key, value in reference_detect.state_dict().items()
    }
    per_variant: dict[str, Any] = {}
    for variant, yolo_model in models.items():
        network = yolo_model.model
        state = state_shape_signature(network, prefix="model.")
        backbone_state = {
            key: value
            for key, value in state.items()
            if int(key.split(".")[1]) <= BACKBONE_LAST_INDEX
        }
        yaml_backbone = read_yaml(variant_config(variant)["yaml_path"])["backbone"]
        baseline_yaml_backbone = read_yaml(
            variant_config("a_base")["yaml_path"]
        )["backbone"]
        current_detect_index = detect_index(network)
        current_detect = network.model[current_detect_index]
        per_variant[variant] = {
            "layer_structure_matches": layer_signature(
                network,
                stop=BACKBONE_LAST_INDEX + 1,
            )
            == reference_layers,
            "state_keys_and_shapes_match": backbone_state
            == reference_backbone_state,
            "yaml_backbone_matches": yaml_backbone == baseline_yaml_backbone,
            "detect_type_matches": current_detect.__class__
            is reference_detect.__class__,
            "detect_state_shapes_match": {
                key: list(value.shape)
                for key, value in current_detect.state_dict().items()
            }
            == reference_detect_state,
            "detect_has_three_inputs": len(current_detect.f) == 3,
            "differences_confined_to_neck": (
                layer_signature(network, stop=BACKBONE_LAST_INDEX + 1)
                == reference_layers
                and current_detect.__class__ is reference_detect.__class__
                and {
                    key: list(value.shape)
                    for key, value in current_detect.state_dict().items()
                }
                == reference_detect_state
            ),
        }
    return {
        "per_variant": per_variant,
        "all_checks_passed": all(
            all(checks.values()) for checks in per_variant.values()
        ),
    }


def gradient_check(
    yolo_model,
    *,
    imgsz: int = 640,
    batch: int = 2,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Backpropagate through every training output and audit gradients."""

    network = yolo_model.model.to(device).train()
    network.zero_grad(set_to_none=True)
    generator = torch.Generator(device="cpu").manual_seed(29)
    x = torch.randn(batch, 3, imgsz, imgsz, generator=generator).to(device)
    output = network(x)
    tensors = [
        tensor
        for tensor in iter_tensors(output)
        if tensor.is_floating_point() and tensor.requires_grad
    ]
    if not tensors:
        raise RuntimeError("Training forward returned no differentiable tensors.")
    loss = sum(tensor.float().square().mean() for tensor in tensors)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f"Non-finite synthetic loss: {loss.item()}.")
    loss.backward()

    missing = []
    nonfinite = []
    zero = []
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        if not bool(parameter.grad.detach().abs().sum() > 0):
            zero.append(name)
    return {
        "loss": float(loss.detach().cpu()),
        "differentiable_output_tensors": len(tensors),
        "trainable_parameter_tensors": sum(
            parameter.requires_grad for parameter in network.parameters()
        ),
        "missing_gradient_keys": missing,
        "nonfinite_gradient_keys": nonfinite,
        "zero_gradient_keys": zero,
        "all_trainable_parameters_have_gradients": not missing,
        "all_gradients_finite": not nonfinite,
    }


def window_padding_check() -> dict[str, Any]:
    from custom_modules.ascgd import (
        ChannelCrossAttention,
        WindowCrossAttention,
        window_partition,
        window_reverse,
    )

    generator = torch.Generator(device="cpu").manual_seed(31)
    x = torch.randn(2, 128, 37, 45, generator=generator, requires_grad=True)
    windows, meta = window_partition(x, 8)
    restored = window_reverse(windows, meta, 8)
    partition_exact = bool(torch.equal(x.detach(), restored.detach()))
    spatial = WindowCrossAttention(128, 128).train()
    channel = ChannelCrossAttention(128, 128).train()
    output = spatial(x, x) + channel(x, x)
    loss = output.float().square().mean()
    loss.backward()
    parameters = list(spatial.parameters()) + list(channel.parameters())
    return {
        "input_shape": list(x.shape),
        "window_shape": list(windows.shape),
        "restored_shape": list(restored.shape),
        "partition_reverse_exact": partition_exact,
        "nonstandard_attention_shape": list(output.shape),
        "nonstandard_attention_finite": bool(torch.isfinite(output).all()),
        "input_gradient_finite": x.grad is not None
        and bool(torch.isfinite(x.grad).all()),
        "parameter_gradients_present": all(
            parameter.grad is not None for parameter in parameters if parameter.requires_grad
        ),
        "positive_channel_temperature": bool(
            (channel.positive_temperature.detach() > 0).all()
        ),
    }


def cuda_amp_check(variant: str = "e_full", imgsz: int = 640) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "status": "not_run",
            "reason": "torch.cuda.is_available() is false on this host",
        }
    model = build_model(variant).model.cuda().train()
    model.zero_grad(set_to_none=True)
    x = torch.randn(1, 3, imgsz, imgsz, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(x)
        tensors = [
            tensor
            for tensor in iter_tensors(output)
            if tensor.is_floating_point() and tensor.requires_grad
        ]
        loss = sum(tensor.float().square().mean() for tensor in tensors)
    loss.backward()
    return {
        "available": True,
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "loss": float(loss.detach().cpu()),
        "all_gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
    }


def load_source_model(weights: str | Path):
    register_modules()
    from ultralytics import YOLO

    path = resolve_pretrained_weights(weights)
    return YOLO(str(path), verbose=False), path


def resolve_pretrained_weights(weights: str | Path) -> Path:
    """Resolve a checkpoint, downloading only the named official yolo11n.pt."""

    requested = Path(weights).expanduser()
    path = requested.resolve()
    if path.is_file():
        return path
    if requested.name != "yolo11n.pt":
        raise FileNotFoundError(f"Pretrained weights not found: {path}")
    register_modules()
    from ultralytics import YOLO

    official = YOLO("yolo11n.pt", verbose=False)
    downloaded = Path(official.ckpt_path).expanduser().resolve()
    if not downloaded.is_file():
        raise FileNotFoundError(
            f"Ultralytics did not produce the official checkpoint: {downloaded}"
        )
    return downloaded


def _layer_key(key: str) -> tuple[int, str] | None:
    parts = key.split(".", 2)
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    return int(parts[1]), parts[2]


def transfer_weights(
    target_yolo,
    weights: str | Path = DEFAULT_WEIGHTS,
    *,
    source_is_inception: bool = False,
    apply: bool = True,
) -> dict[str, Any]:
    """Inherit backbone and Detect by semantic layer, never by shape coercion."""

    source_yolo, source_path = load_source_model(weights)
    source = source_yolo.model
    target = target_yolo.model
    source_state = source.state_dict()
    target_state = target.state_dict()
    source_detect = detect_index(source)
    target_detect = detect_index(target)
    target_is_baseline = target_detect == source_detect and len(target.model) == len(source.model)

    parameter_keys = {name for name, _parameter in target.named_parameters()}
    inherited: dict[str, str] = {}
    shape_mismatches: list[dict[str, Any]] = []
    candidate_missing: list[str] = []
    updated = dict(target_state)

    for target_key, target_tensor in target_state.items():
        parsed = _layer_key(target_key)
        if parsed is None:
            continue
        target_layer, suffix = parsed
        source_key: str | None = None
        if target_layer <= BACKBONE_LAST_INDEX:
            source_key = target_key
        elif target_layer == target_detect:
            source_key = f"model.{source_detect}.{suffix}"
        elif target_is_baseline:
            source_key = target_key

        if source_key is None:
            continue
        source_tensor = source_state.get(source_key)
        if source_tensor is None:
            candidate_missing.append(target_key)
            continue
        if source_tensor.shape != target_tensor.shape:
            shape_mismatches.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue
        updated[target_key] = source_tensor.detach().to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        )
        inherited[target_key] = source_key

    if apply:
        target.load_state_dict(updated, strict=True)

    inherited_parameter_keys = parameter_keys.intersection(inherited)
    target_parameter_elements = sum(
        parameter.numel() for parameter in target.parameters()
    )
    inherited_parameter_elements = sum(
        target_state[key].numel() for key in inherited_parameter_keys
    )
    backbone_parameter_keys = {
        key
        for key in parameter_keys
        if (_layer_key(key) or (999, ""))[0] <= BACKBONE_LAST_INDEX
    }
    detect_parameter_keys = {
        key
        for key in parameter_keys
        if (_layer_key(key) or (-1, ""))[0] == target_detect
    }
    random_parameter_keys = sorted(parameter_keys - inherited_parameter_keys)
    random_modules = sorted(
        {
            ".".join(key.split(".")[:3])
            for key in random_parameter_keys
            if len(key.split(".")) >= 3
        }
    )
    return {
        "source_weights": portable_path(source_path),
        "source_sha256": sha256_file(source_path),
        "source_is_inception_best_debug": source_is_inception,
        "source_detect_index": source_detect,
        "target_detect_index": target_detect,
        "total_state_tensors": len(target_state),
        "inherited_state_tensors": len(inherited),
        "total_parameter_tensors": len(parameter_keys),
        "inherited_parameter_tensors": len(inherited_parameter_keys),
        "target_parameter_elements": target_parameter_elements,
        "inherited_parameter_elements": inherited_parameter_elements,
        "parameter_element_inheritance_ratio": inherited_parameter_elements
        / target_parameter_elements,
        "backbone_parameter_elements": sum(
            target_state[key].numel() for key in backbone_parameter_keys
        ),
        "inherited_backbone_parameter_elements": sum(
            target_state[key].numel()
            for key in backbone_parameter_keys.intersection(inherited)
        ),
        "backbone_parameter_inheritance_ratio": (
            sum(
                target_state[key].numel()
                for key in backbone_parameter_keys.intersection(inherited)
            )
            / sum(target_state[key].numel() for key in backbone_parameter_keys)
        ),
        "detect_parameter_elements": sum(
            target_state[key].numel() for key in detect_parameter_keys
        ),
        "inherited_detect_parameter_elements": sum(
            target_state[key].numel()
            for key in detect_parameter_keys.intersection(inherited)
        ),
        "detect_parameter_inheritance_ratio": (
            sum(
                target_state[key].numel()
                for key in detect_parameter_keys.intersection(inherited)
            )
            / sum(target_state[key].numel() for key in detect_parameter_keys)
        ),
        "random_initialized_parameter_keys": random_parameter_keys,
        "random_initialized_modules": random_modules,
        "shape_mismatches": shape_mismatches,
        "candidate_missing": candidate_missing,
        "matched_source_to_target": [
            {"source": source_key, "target": target_key}
            for target_key, source_key in sorted(inherited.items())
        ],
        "forced_crop_repeat_or_pad": False,
        "applied": apply,
    }


def save_initialized_model(
    variant: str,
    weights: str | Path,
    output: str | Path,
    *,
    source_is_inception: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Build, transfer, and save a real checkpoint used by Trainer."""

    target = build_model(variant, seed=seed)
    transfer = transfer_weights(
        target,
        weights,
        source_is_inception=source_is_inception,
        apply=True,
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target.save(output)
    return {
        "variant": variant,
        "model_yaml": str(variant_config(variant)["yaml_path"]),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "git_commit": git_commit(),
        "ultralytics_version": require_ultralytics_version(),
        "transfer": transfer,
    }


def runtime_versions() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": require_ultralytics_version(),
    }


def _dataset_root(data_yaml: Path, config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("path") or data_yaml.parent)).expanduser()
    return (
        configured.resolve()
        if configured.is_absolute()
        else (data_yaml.parent / configured).resolve()
    )


def _dataset_split_files(value: Any, root: Path) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for item in values:
        path = Path(str(item)).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path.is_dir():
            images.extend(
                sorted(
                    file
                    for file in path.rglob("*")
                    if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip()).expanduser()
                images.append(
                    image.resolve()
                    if image.is_absolute()
                    else (root / image).resolve()
                )
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise FileNotFoundError(f"Dataset split path is missing or unsupported: {path}")
    return images


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if indices:
        parts[indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def audit_dataset(
    data: str | Path,
    *,
    expected: dict[str, int] | None = None,
    require_single_class: bool = True,
) -> dict[str, Any]:
    """Validate data.yaml and report split/image/label count differences."""

    data_yaml = Path(data).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    config = read_yaml(data_yaml)
    nc = int(config.get("nc", len(config.get("names", []))))
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    if require_single_class and nc != 1:
        raise ValueError(f"ASCGD formal training requires nc=1, found nc={nc}.")
    root = _dataset_root(data_yaml, config)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    expected = expected or {"train": 2582, "val": 842, "test": 874}
    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        if split not in config or config[split] in (None, ""):
            splits[split] = {
                "present": False,
                "images": 0,
                "labels": 0,
                "expected_images": expected.get(split),
                "difference": None,
            }
            continue
        images = _dataset_split_files(config[split], root)
        labels = [_label_for_image(image) for image in images]
        label_count = sum(label.is_file() for label in labels)
        expected_count = expected.get(split)
        splits[split] = {
            "present": True,
            "images": len(images),
            "labels": label_count,
            "missing_labels": len(images) - label_count,
            "expected_images": expected_count,
            "difference": (
                len(images) - expected_count
                if expected_count is not None
                else None
            ),
        }
    if not splits["train"]["present"] or not splits["val"]["present"]:
        raise RuntimeError("Dataset YAML must contain train and val splits.")
    if not splits["train"]["images"] or not splits["val"]["images"]:
        raise RuntimeError("Train and val splits must each contain images.")
    return {
        "data_yaml": str(data_yaml),
        "root": str(root),
        "nc": nc,
        "names": names,
        "splits": splits,
        "expected_counts_match": all(
            not values["present"] or values["difference"] in (None, 0)
            for values in splits.values()
        ),
    }
