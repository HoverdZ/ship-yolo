"""Reproducible helpers for the P2/P3 convolution-screening experiments.

The public training path deliberately stays in the calling notebook kernel:
the notebook invokes Ultralytics ``YOLO.train`` directly. This module handles
dataset localization, model construction, pretrained-weight transfer,
structure checks, handoff verification, and safe run-name selection only.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_VERSION = "8.4.92"

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "C0_yolo11n_official": {
        "name": "YOLO11n official baseline",
        "yaml": "experiments/conv_screening_v1/C0_yolo11n_official.yaml",
        "module": "C3k2",
        "custom_indices": [],
        "run_name": "yolo11n_official_baseline_convscreen_640",
    },
    "C1_pconv_p23": {
        "name": "YOLO11n-PConv-P2P3",
        "yaml": "experiments/conv_screening_v1/C1_yolo11n_pconv_p23.yaml",
        "module": "C3k2_PConv",
        "custom_indices": [2, 4],
        "run_name": "yolo11n_pconv_p23_640",
    },
    "C2_lskconv_p23": {
        "name": "YOLO11n-LSKConv-P2P3",
        "yaml": "experiments/conv_screening_v1/C2_yolo11n_lskconv_p23.yaml",
        "module": "C3k2_LSKConv",
        "custom_indices": [2, 4],
        "run_name": "yolo11n_lskconv_p23_640",
    },
}

CUSTOM_MODULE_NAMES = {
    "C3k2_PConv",
    "C3k2_LSKConv",
}
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


@dataclass(frozen=True)
class ConvScreeningConfig:
    """Fixed comparison settings plus runtime paths for one experiment."""

    experiment_id: str
    drive_data_root: str = "/content/drive/MyDrive/ship_detection/data"
    local_data_root: str = "/content/ship_detection/data"
    drive_runs_root: str = "/content/drive/MyDrive/ship_detection/runs"
    epochs: int = 150
    imgsz: int = 640
    batch: int = 8
    workers: int = 2
    seed: int = 0
    device: int | str = 0
    cache: str = "disk"
    deterministic: bool = False
    save_period: int = 10
    copy_workers: int = 16

    def __post_init__(self) -> None:
        if self.experiment_id not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment_id: {self.experiment_id}")
        fixed = {
            "epochs": 150,
            "imgsz": 640,
            "batch": 8,
            "workers": 2,
            "seed": 0,
            "cache": "disk",
            "deterministic": False,
            "save_period": 10,
        }
        changed = {
            key: (getattr(self, key), expected)
            for key, expected in fixed.items()
            if getattr(self, key) != expected
        }
        if changed:
            raise ValueError(
                "Controlled-comparison settings are fixed; rejected overrides: "
                f"{changed}"
            )
        if self.copy_workers < 1:
            raise ValueError("copy_workers must be positive.")

    @property
    def spec(self) -> dict[str, Any]:
        return EXPERIMENTS[self.experiment_id]

    @property
    def model_yaml(self) -> Path:
        return ROOT / self.spec["yaml"]

    @property
    def base_run_name(self) -> str:
        return str(self.spec["run_name"])

    @property
    def local_yaml(self) -> Path:
        return Path(self.local_data_root).parent / "data_local.yaml"


def _write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def _find_data_yaml(source_root: Path) -> Path:
    preferred = source_root / "data.yaml"
    if preferred.is_file():
        return preferred
    candidates = sorted(source_root.glob("*.yaml")) + sorted(source_root.glob("*.yml"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No data.yaml was found under the Drive dataset root: {source_root}"
        )
    raise RuntimeError(
        "Multiple dataset YAML files were found; keep data.yaml as the unambiguous "
        f"training definition. Found: {[path.name for path in candidates]}"
    )


def _copy_one(source: Path, destination: Path) -> tuple[int, bool]:
    """Copy one file with a same-size fast path and post-copy size check."""

    size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        return size, False
    import shutil

    shutil.copyfile(source, destination)
    copied_size = destination.stat().st_size
    if copied_size != size:
        raise IOError(
            f"Size mismatch after copy: {source} -> {destination} "
            f"({size} != {copied_size})"
        )
    return size, True


def copy_dataset_to_local(config: ConvScreeningConfig) -> dict[str, Any]:
    """Copy Drive data with concurrent ``shutil.copyfile`` and live progress."""

    source_root = Path(config.drive_data_root)
    destination_root = Path(config.local_data_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Drive dataset root does not exist: {source_root}")
    source_yaml = _find_data_yaml(source_root)

    print(f"Scanning dataset files: {source_root}", flush=True)
    source_files: list[Path] = []
    for current_root, _directories, filenames in os.walk(source_root):
        current = Path(current_root)
        source_files.extend(current / filename for filename in filenames)
        if len(source_files) and len(source_files) % 500 == 0:
            print(
                f"\rDiscovered at least {len(source_files):,} files...",
                end="",
                flush=True,
            )
    source_files.sort()
    print(f"\rDiscovered {len(source_files):,} files.          ", flush=True)
    if not source_files:
        raise FileNotFoundError(f"No dataset files were found under {source_root}.")

    jobs = [
        (source, destination_root / source.relative_to(source_root))
        for source in source_files
    ]
    started = time.perf_counter()
    processed_bytes = copied_bytes = copied_files = 0
    print(
        f"Copying with {config.copy_workers} threads; existing same-size files "
        "are verified and skipped.",
        flush=True,
    )
    with (
        tqdm(
            total=len(jobs),
            desc="Dataset files",
            unit="file",
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stdout,
        ) as file_bar,
        tqdm(
            total=None,
            desc="Processed bytes",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stdout,
        ) as byte_bar,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=config.copy_workers
        ) as executor,
    ):
        future_to_source = {
            executor.submit(_copy_one, source, destination): source
            for source, destination in jobs
        }
        for future in concurrent.futures.as_completed(future_to_source):
            source = future_to_source[future]
            try:
                size, copied = future.result()
            except Exception as error:
                raise IOError(f"Dataset copy failed for {source}: {error}") from error
            processed_bytes += size
            copied_files += int(copied)
            copied_bytes += size if copied else 0
            file_bar.update(1)
            byte_bar.update(size)
            file_bar.set_postfix(
                copied=copied_files,
                workers=config.copy_workers,
                GiB=f"{processed_bytes / 1024**3:.2f}",
                refresh=False,
            )

    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    missing = [
        key for key in ("train", "val", "names")
        if key not in payload
    ]
    if missing:
        raise KeyError(f"Dataset YAML is missing required keys: {missing}")
    names = payload["names"]
    if not isinstance(names, (list, dict)) or not names:
        raise ValueError("Dataset names must be a non-empty list or mapping.")
    nc = int(payload.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"Dataset nc={nc} differs from len(names)={len(names)}.")

    local_payload = dict(payload)
    local_payload["path"] = str(destination_root)
    local_payload["nc"] = nc
    config.local_yaml.parent.mkdir(parents=True, exist_ok=True)
    config.local_yaml.write_text(
        yaml.safe_dump(local_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    report = {
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "source_yaml": str(source_yaml),
        "local_yaml": str(config.local_yaml),
        "files_processed": len(jobs),
        "bytes_processed": processed_bytes,
        "files_copied": copied_files,
        "bytes_copied": copied_bytes,
        "copy_workers": config.copy_workers,
        "elapsed_seconds": elapsed,
        "nc": nc,
        "splits_preserved": [
            split for split in ("train", "val", "test")
            if split in local_payload
        ],
        "fixed_count_comparison_performed": False,
    }
    _write_json(
        config.local_yaml.with_name("dataset_copy_report.json"),
        report,
    )
    print(
        f"Dataset ready: {len(jobs):,} files / "
        f"{processed_bytes / 1024**3:.2f} GiB processed in {elapsed:.1f}s; "
        f"{copied_files:,} files actually copied.",
        flush=True,
    )
    print("Local data YAML:", config.local_yaml, flush=True)
    return report


def _dataset_nc(local_yaml: Path) -> int:
    payload = yaml.safe_load(local_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if not isinstance(names, (list, dict)) or not names:
        raise ValueError(f"Invalid names in {local_yaml}")
    nc = int(payload.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"Dataset nc={nc} differs from len(names)={len(names)}.")
    return nc


def register_modules() -> None:
    """Register repository modules without editing Ultralytics site-packages."""

    from custom_modules.register import register_conv_screening_modules

    register_conv_screening_modules()


def _build_model(model_yaml: Path, nc: int):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    wrapper = YOLO(str(model_yaml), verbose=False)
    if int(wrapper.model.model[-1].nc) != nc:
        wrapper.model = DetectionModel(
            cfg=str(model_yaml),
            ch=3,
            nc=nc,
            verbose=False,
        )
        wrapper.task = "detect"
    return wrapper


def transfer_official_pretrained(
    target,
    official_weights: str | Path = "yolo11n.pt",
) -> dict[str, Any]:
    """Apply and verify all exact-name, exact-shape official tensors."""

    from ultralytics import YOLO

    source = YOLO(str(official_weights), verbose=False)
    source_state = source.model.float().state_dict()
    target_state = target.model.state_dict()
    compatible = {
        key: source_state[key].detach().cpu()
        for key, value in target_state.items()
        if key in source_state
        and tuple(value.shape) == tuple(source_state[key].shape)
    }
    result = target.model.load_state_dict(compatible, strict=False)
    loaded_state = target.model.state_dict()
    verification_failures = [
        key
        for key, expected in compatible.items()
        if not torch.equal(loaded_state[key].detach().cpu(), expected)
    ]
    unmatched_target = sorted(set(target_state) - set(compatible))
    out_of_scope_unmatched = [
        key
        for key in unmatched_target
        if not (
            (
                (
                    key.startswith("model.2.m.")
                    or key.startswith("model.4.m.")
                )
                and ".cv2." in key
            )
            # COCO pretraining has nc=80. A one-class dataset necessarily
            # rebuilds Detect.cv3, while the box-regression branch still loads.
            or key.startswith("model.23.cv3.")
        )
    ]
    p2_p3_cv1_keys = [
        key
        for key in target_state
        if (
            key.startswith("model.2.m.")
            or key.startswith("model.4.m.")
        )
        and ".cv1." in key
    ]
    p2_p3_cv1_missing = sorted(set(p2_p3_cv1_keys) - set(compatible))
    parameter_names = set(dict(target.model.named_parameters()))
    total_parameter_elements = sum(value.numel() for value in target.model.parameters())
    loaded_parameter_elements = sum(
        target_state[key].numel()
        for key in compatible
        if key in parameter_names
    )
    report = {
        "official_weights": str(official_weights),
        "method": "same_name_same_shape",
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
        "loaded_tensors": len(compatible),
        "loaded_total": f"{len(compatible)}/{len(target_state)}",
        "tensor_inheritance_ratio": len(compatible) / len(target_state),
        "target_parameter_elements": total_parameter_elements,
        "loaded_parameter_elements": loaded_parameter_elements,
        "parameter_element_inheritance_ratio": (
            loaded_parameter_elements / total_parameter_elements
        ),
        "loaded_target_keys": sorted(compatible),
        "unmatched_target_keys": unmatched_target,
        "out_of_scope_unmatched_target_keys": out_of_scope_unmatched,
        "p2_p3_cv1_expected_keys": sorted(p2_p3_cv1_keys),
        "p2_p3_cv1_missing_keys": p2_p3_cv1_missing,
        "missing_after_load": list(result.missing_keys),
        "unexpected_after_load": list(result.unexpected_keys),
        "verification_failures": verification_failures,
    }
    report["passed"] = not any(
        (
            result.unexpected_keys,
            verification_failures,
            out_of_scope_unmatched,
            p2_p3_cv1_missing,
        )
    )
    if not report["passed"]:
        raise RuntimeError(
            "Official pretrained transfer failed its scope audit: "
            f"verification={len(verification_failures)}, "
            f"out_of_scope_unmatched={len(out_of_scope_unmatched)}, "
            f"P2/P3_cv1_missing={len(p2_p3_cv1_missing)}"
        )
    return report


def audit_structure(
    config: ConvScreeningConfig,
    wrapper,
) -> dict[str, Any]:
    """Verify the official baseline or the scoped P2/P3 operator change."""

    from ultralytics.nn.modules import Conv

    layers = wrapper.model.model
    layer_types = [type(layer).__name__ for layer in layers]
    expected_module = str(config.spec["module"])
    custom_indices = [
        index for index, name in enumerate(layer_types)
        if name in CUSTOM_MODULE_NAMES
    ]
    selected = [layers[2], layers[4]]
    expected_custom_indices = list(config.spec["custom_indices"])
    cv1_checks: list[dict[str, Any]] = []
    for layer_index, layer in zip((2, 4), selected, strict=True):
        for block_index, block in enumerate(layer.m):
            conv = getattr(block.cv1, "conv", None)
            cv1_checks.append(
                {
                    "layer": layer_index,
                    "block": block_index,
                    "class": type(block.cv1).__name__,
                    "is_ultralytics_conv": isinstance(block.cv1, Conv),
                    "kernel_size": list(conv.kernel_size) if conv is not None else None,
                    "stride": list(conv.stride) if conv is not None else None,
                    "input_channels": int(conv.in_channels) if conv is not None else None,
                    "output_channels": int(conv.out_channels) if conv is not None else None,
                    "official_half_expansion": (
                        conv is not None
                        and int(conv.in_channels) == 2 * int(conv.out_channels)
                    ),
                }
            )
    downsample_checks: list[dict[str, Any]] = []
    for index in (0, 1, 3, 5, 7):
        layer = layers[index]
        conv = getattr(layer, "conv", None)
        downsample_checks.append(
            {
                "layer": index,
                "class": type(layer).__name__,
                "is_ultralytics_conv": isinstance(layer, Conv),
                "kernel_size": list(conv.kernel_size) if conv is not None else None,
                "stride": list(conv.stride) if conv is not None else None,
            }
        )
    detect = layers[-1]
    checks = {
        "selected_module_at_P2_P3": (
            layer_types[2] == expected_module
            and layer_types[4] == expected_module
        ),
        "custom_scope_as_declared": custom_indices == expected_custom_indices,
        "first_3x3_preserved": all(
            item["is_ultralytics_conv"]
            and item["kernel_size"] == [3, 3]
            and item["stride"] == [1, 1]
            and item["official_half_expansion"]
            for item in cv1_checks
        ),
        "downsampling_unchanged": all(
            item["is_ultralytics_conv"]
            and item["kernel_size"] == [3, 3]
            and item["stride"] == [2, 2]
            for item in downsample_checks
        ),
        "detect_from_unchanged": list(detect.f) == [16, 19, 22],
        "detect_strides_unchanged": [
            float(value) for value in wrapper.model.stride
        ] == [8.0, 16.0, 32.0],
    }
    report = {
        "experiment_id": config.experiment_id,
        "model_yaml": str(config.model_yaml),
        "layer_types": layer_types,
        "custom_indices": custom_indices,
        "expected_custom_indices": expected_custom_indices,
        "expected_module": expected_module,
        "cv1_checks": cv1_checks,
        "downsample_checks": downsample_checks,
        "detect_from": list(detect.f),
        "detect_strides": [
            float(value) for value in wrapper.model.stride
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not report["passed"]:
        raise AssertionError(f"Structure audit failed: {report}")
    return report


def _flatten_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _flatten_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_tensors(child)


def cpu_forward_backward(wrapper, imgsz: int = 64, seed: int = 0) -> dict[str, Any]:
    """Run a deterministic CPU forward/backward smoke check."""

    network = wrapper.model.cpu()
    was_training = network.training
    network.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        generator=generator,
        requires_grad=True,
    )
    output = network(image)
    tensors = list(_flatten_tensors(output))
    loss = sum(tensor.float().square().mean() for tensor in tensors)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in network.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    report = {
        "imgsz": imgsz,
        "output_tensors": len(tensors),
        "outputs_finite": bool(tensors)
        and all(torch.isfinite(tensor).all().item() for tensor in tensors),
        "input_gradient_finite": image.grad is not None
        and torch.isfinite(image.grad).all().item(),
        "parameter_gradients": len(gradients),
        "parameter_gradients_finite": bool(gradients)
        and all(torch.isfinite(gradient).all().item() for gradient in gradients),
    }
    report["passed"] = all(
        report[key]
        for key in (
            "outputs_finite",
            "input_gradient_finite",
            "parameter_gradients_finite",
        )
    )
    network.zero_grad(set_to_none=True)
    network.train(was_training)
    if not report["passed"]:
        raise RuntimeError(f"CPU forward/backward check failed: {report}")
    return report


def prepare_model(
    config: ConvScreeningConfig,
    official_weights: str | Path = "yolo11n.pt",
    *,
    run_cpu_check: bool = False,
):
    """Build, initialize, and audit one controlled experiment."""

    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected ultralytics=={ULTRALYTICS_VERSION}, "
            f"found {ultralytics.__version__}"
        )
    if not config.local_yaml.is_file():
        raise FileNotFoundError(
            f"Local dataset YAML is missing; run copy_dataset_to_local first: "
            f"{config.local_yaml}"
        )
    register_modules()
    torch.manual_seed(config.seed)
    wrapper = _build_model(config.model_yaml, _dataset_nc(config.local_yaml))
    structure = audit_structure(config, wrapper)
    transfer = transfer_official_pretrained(wrapper, official_weights)
    smoke = cpu_forward_backward(wrapper, seed=config.seed) if run_cpu_check else None

    # Ultralytics 8.4.92 forwards an in-memory initialized model to its trainer
    # only when YOLO.ckpt is truthy. ``pretrained=True`` must also be supplied
    # by the notebook's direct model.train(...) call.
    wrapper.ckpt = {
        "model": wrapper.model,
        "epoch": -1,
        "optimizer": None,
        "conv_screening_pretrained_transfer": True,
    }
    return {
        "model": wrapper,
        "structure": structure,
        "transfer": transfer,
        "cpu_smoke": smoke,
    }


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        try:
            digest.update(tensor.numpy().tobytes())
        except TypeError:
            digest.update(tensor.float().numpy().tobytes())
    return digest.hexdigest()


def _unwrap_training_model(model):
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def install_trainer_handoff_guard(
    wrapper,
    transfer_report: dict[str, Any],
    report_path: str | Path,
    expected_run_dir: str | Path,
) -> None:
    """Raise before epoch 1 if the trainer discarded initialized weights."""

    expected_state = {
        key: value.detach().cpu().clone()
        for key, value in wrapper.model.state_dict().items()
    }
    official_keys = set(transfer_report["loaded_target_keys"])
    output = Path(report_path)
    expected_directory = Path(expected_run_dir).resolve()

    def verify_run_directory(trainer) -> None:
        actual_directory = Path(trainer.save_dir).resolve()
        if actual_directory != expected_directory:
            raise RuntimeError(
                "Ultralytics save directory drifted from the collision-safe "
                f"selection: {actual_directory} != {expected_directory}"
            )

    def verify_handoff(trainer) -> None:
        actual_state = _unwrap_training_model(trainer.model).state_dict()
        missing = sorted(set(expected_state) - set(actual_state))
        unexpected = sorted(set(actual_state) - set(expected_state))
        shape_mismatches = sorted(
            key
            for key in set(expected_state).intersection(actual_state)
            if tuple(expected_state[key].shape) != tuple(actual_state[key].shape)
        )
        value_mismatches = sorted(
            key
            for key in set(expected_state).intersection(actual_state)
            if key not in shape_mismatches
            and not torch.equal(
                expected_state[key],
                actual_state[key].detach().cpu(),
            )
        )
        mismatches = set(missing) | set(shape_mismatches) | set(value_mismatches)
        official_mismatches = sorted(official_keys.intersection(mismatches))
        report = {
            "expected_tensors": len(expected_state),
            "actual_tensors": len(actual_state),
            "exact_tensors": (
                len(expected_state)
                - len(missing)
                - len(shape_mismatches)
                - len(value_mismatches)
            ),
            "official_pretrained_tensors_expected": len(official_keys),
            "official_pretrained_tensors_preserved": (
                len(official_keys) - len(official_mismatches)
            ),
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatches": shape_mismatches,
            "value_mismatches": value_mismatches,
            "official_pretrained_mismatches": official_mismatches,
            "expected_state_sha256": _state_sha256(expected_state),
            "actual_state_sha256": _state_sha256(actual_state),
        }
        report["passed"] = not any(
            (missing, unexpected, shape_mismatches, value_mismatches)
        )
        _write_json(output, report)
        if not report["passed"]:
            raise RuntimeError(
                "Ultralytics trainer discarded or altered initialized weights "
                "before epoch 1; training was stopped. "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"shape={len(shape_mismatches)}, values={len(value_mismatches)}"
            )
        print(
            "Trainer handoff exact tensors:",
            f"{report['exact_tensors']}/{report['expected_tensors']}",
            flush=True,
        )
        print(
            "Official pretrained tensors preserved:",
            f"{report['official_pretrained_tensors_preserved']}/"
            f"{report['official_pretrained_tensors_expected']}",
            flush=True,
        )

    wrapper.add_callback("on_pretrain_routine_start", verify_run_directory)
    wrapper.add_callback("on_pretrain_routine_end", verify_handoff)


def _checkpoint_status(path: Path, epochs: int) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        return {
            "valid_resume": False,
            "complete": False,
            "error": f"{type(error).__name__}: {error}",
        }
    if not isinstance(checkpoint, dict):
        return {
            "valid_resume": False,
            "complete": False,
            "error": f"Checkpoint type is {type(checkpoint).__name__}, expected dict.",
        }
    epoch = int(checkpoint.get("epoch", -1))
    train_args = checkpoint.get("train_args") or {}
    complete = epoch + 1 >= epochs
    valid_resume = (
        epoch >= 0
        and not complete
        and checkpoint.get("optimizer") is not None
        and isinstance(train_args, dict)
        and bool(train_args)
    )
    return {
        "valid_resume": valid_resume,
        "complete": complete,
        "epoch": epoch,
        "has_optimizer": checkpoint.get("optimizer") is not None,
        "has_train_args": isinstance(train_args, dict) and bool(train_args),
        "error": None,
    }


def resolve_run_state(config: ConvScreeningConfig) -> dict[str, Any]:
    """Resume a valid interrupted run or allocate a collision-free fresh name."""

    project = Path(config.drive_runs_root)
    project.mkdir(parents=True, exist_ok=True)
    base_name = config.base_run_name
    base_dir = project / base_name
    last = base_dir / "weights" / "last.pt"
    if last.is_file():
        status = _checkpoint_status(last, config.epochs)
        if status["valid_resume"]:
            return {
                "mode": "resume",
                "run_name": base_name,
                "run_dir": str(base_dir),
                "resume_checkpoint": str(last),
                "checkpoint": status,
            }
    elif not base_dir.exists() or not any(base_dir.iterdir()):
        return {
            "mode": "new",
            "run_name": base_name,
            "run_dir": str(base_dir),
            "resume_checkpoint": None,
            "reason": "base run directory is available",
        }

    # Preserve every old artifact. A completed, invalid, or partial residue
    # receives a new independent retry name instead of causing FileExistsError.
    retry = 1
    while True:
        run_name = f"{base_name}_retry{retry}"
        run_dir = project / run_name
        if not run_dir.exists() or not any(run_dir.iterdir()):
            return {
                "mode": "new",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "resume_checkpoint": None,
                "reason": (
                    f"{base_dir} already contains non-resumable or completed "
                    "artifacts; they were preserved"
                ),
            }
        retry += 1


def save_preflight_reports(
    config: ConvScreeningConfig,
    run_state: dict[str, Any],
    prepared: dict[str, Any],
    copy_report: dict[str, Any],
) -> Path:
    """Persist exact preflight evidence next to the future training run."""

    run_dir = Path(run_state["run_dir"])
    preflight = run_dir / "preflight"
    preflight.mkdir(parents=True, exist_ok=True)
    _write_json(preflight / "config.json", asdict(config))
    _write_json(preflight / "run_state.json", run_state)
    _write_json(preflight / "dataset_copy_report.json", copy_report)
    _write_json(preflight / "structure_audit.json", prepared["structure"])
    _write_json(preflight / "pretrained_transfer_report.json", prepared["transfer"])
    return preflight


def best_metrics(run_dir: str | Path) -> dict[str, Any]:
    """Read the best validation epoch from Ultralytics results.csv."""

    path = Path(run_dir) / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    normalized = [
        {key.strip(): value for key, value in row.items()}
        for row in rows
    ]
    columns = list(normalized[0])

    def column(suffix: str) -> str:
        return next(name for name in columns if name.endswith(suffix))

    map_column = column("mAP50-95(B)")
    best = max(normalized, key=lambda row: float(row[map_column]))
    report = {
        "selection_rule": "maximum validation mAP50-95",
        # Ultralytics 8.4.92 writes human-facing epoch numbers (1..N) to
        # results.csv, so no zero-based conversion is required here.
        "best_epoch": int(float(best["epoch"])),
        "precision": float(best[column("precision(B)")]),
        "recall": float(best[column("recall(B)")]),
        "map50": float(best[column("mAP50(B)")]),
        "map50_95": float(best[map_column]),
        "results_csv": str(path),
    }
    _write_json(Path(run_dir) / "best_metrics.json", report)
    return report


__all__ = [
    "ConvScreeningConfig",
    "EXPERIMENTS",
    "audit_structure",
    "best_metrics",
    "copy_dataset_to_local",
    "cpu_forward_backward",
    "install_trainer_handoff_guard",
    "prepare_model",
    "register_modules",
    "resolve_run_state",
    "save_preflight_reports",
    "transfer_official_pretrained",
]
