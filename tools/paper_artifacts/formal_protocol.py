"""Shared, foreground-only protocol for the six formal YOLO11n ablations.

The public functions in this module are called directly by Colab notebook
cells. Training is deliberately performed by ``YOLO.train`` in the current
Python kernel; this module never launches training in a subprocess.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import csv
import hashlib
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

import torch
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_VERSION = "8.4.92"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
REQUIRED_SPLITS = ("train", "val", "test")
KNOWN_AUDITED_COUNTS = {
    "train": {"images": 2517, "labels": 2517, "empty_labels": 1222},
    "val": {"images": 839, "labels": 839, "empty_labels": 413},
    "test": {"images": 840, "labels": 840, "empty_labels": 443},
}

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "A0_yolo11n": {
        "model_name": "YOLO11n",
        "yaml": "experiments/formal_ablation_v1/A0_yolo11n.yaml",
        "strides": [8.0, 16.0, 32.0],
        "detect_from": [16, 19, 22],
        "modules": [],
        "specialty": "baseline",
    },
    "A1_inceptiondw": {
        "model_name": "YOLO11n + InceptionDW",
        "yaml": "experiments/formal_ablation_v1/A1_inceptiondw.yaml",
        "strides": [8.0, 16.0, 32.0],
        "detect_from": [16, 19, 22],
        "modules": ["C3k2_InceptionDW"],
        "specialty": "inceptiondw",
    },
    "A2_inceptiondw_dpls": {
        "model_name": "YOLO11n + InceptionDW + DPLS",
        "yaml": "experiments/formal_ablation_v1/A2_inceptiondw_dpls.yaml",
        "strides": [4.0, 8.0, 16.0],
        "detect_from": [14, 17, 20],
        "modules": ["C3k2_InceptionDW", "DySample"],
        "specialty": "dpls",
    },
    "A3_inceptiondw_dpls_scam": {
        "model_name": "YOLO11n + InceptionDW + DPLS + SCAM",
        "yaml": "experiments/formal_ablation_v1/A3_inceptiondw_dpls_scam.yaml",
        "strides": [4.0, 8.0, 16.0],
        "detect_from": [21, 22, 23],
        "modules": ["C3k2_InceptionDW", "DySample", "SCAM"],
        "specialty": "scam",
    },
    "A4_inceptiondw_dpls_scam_vgup": {
        "model_name": "YOLO11n + InceptionDW + DPLS + SCAM + VGUP",
        "yaml": "experiments/formal_ablation_v1/A4_inceptiondw_dpls_scam_vgup.yaml",
        "strides": [4.0, 8.0, 16.0],
        "detect_from": [22, 23, 24],
        "modules": ["VGUPPreprocessor", "C3k2_InceptionDW", "DySample", "SCAM"],
        "specialty": "vgup",
        "shifted_official_mapping": True,
    },
    "A5_inceptiondw_dpls_ca_scam_vgup": {
        "model_name": "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP",
        "yaml": "experiments/formal_ablation_v1/A5_inceptiondw_dpls_ca_scam_vgup.yaml",
        "strides": [4.0, 8.0, 16.0],
        "detect_from": [22, 23, 24],
        "modules": ["VGUPPreprocessor", "C3k2_InceptionDW", "DySample", "CASCAM"],
        "specialty": "ca_scam",
        "shifted_official_mapping": True,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


@dataclass(frozen=True)
class FormalConfig:
    experiment_id: str
    drive_data_yaml: str = "/content/drive/MyDrive/ship_detection/data/data.yaml"
    drive_data_root: str | None = None
    local_data_root: str = "/content/datasets/ship_clean_v1"
    drive_experiment_root: str = "/content/drive/MyDrive/ShipPaper/formal_ablation_v1"
    local_runs_root: str = "/content/formal_runs"
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
    run_training: bool = True
    run_test_evaluation: bool = False
    conf: float = 0.25
    iou: float = 0.7
    tiny_short_side: float = 16.0
    small_short_side: float = 32.0

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
        changed = {key: (getattr(self, key), value) for key, value in fixed.items() if getattr(self, key) != value}
        if changed:
            raise ValueError(f"Formal-comparison settings are fixed; rejected overrides: {changed}")

    @property
    def spec(self) -> dict[str, Any]:
        return EXPERIMENTS[self.experiment_id]

    @property
    def model_yaml(self) -> Path:
        return ROOT / self.spec["yaml"]

    @property
    def run_dir(self) -> Path:
        return Path(self.local_runs_root) / self.experiment_id

    @property
    def drive_dir(self) -> Path:
        return Path(self.drive_experiment_root) / self.experiment_id

    @property
    def protocol_staging_dir(self) -> Path:
        return Path(self.local_runs_root) / ".protocol_staging" / self.experiment_id

    @property
    def local_yaml(self) -> Path:
        return Path(self.local_data_root).parent / "data_runtime_local.yaml"


def require_environment() -> dict[str, Any]:
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected ultralytics=={ULTRALYTICS_VERSION}, found {ultralytics.__version__}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Formal notebooks require Python 3.12.x, found {platform.python_version()}")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    }


def _resolve_split(root: Path, value: Any) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths: list[Path] = []
    for item in values:
        path = Path(str(item))
        paths.append(path if path.is_absolute() else root / path)
    return paths


def resolve_dataset(config: FormalConfig) -> tuple[dict[str, Any], Path, dict[str, list[Path]]]:
    data_yaml = Path(config.drive_data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Drive data YAML does not exist: {data_yaml}")
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    missing = [key for key in REQUIRED_SPLITS if key not in payload]
    if "names" not in payload:
        missing.append("names")
    if missing:
        raise KeyError(f"Dataset YAML is missing required keys: {missing}")
    yaml_parent = data_yaml.parent.resolve()
    configured_path = Path(str(payload.get("path", yaml_parent)))
    candidates: list[Path] = []
    if config.drive_data_root:
        candidates.append(Path(config.drive_data_root))
    if configured_path.is_absolute():
        candidates.append(configured_path)
    else:
        candidates.append((yaml_parent / configured_path).resolve())
    candidates.append(yaml_parent)
    checked: list[dict[str, Any]] = []
    for candidate in dict.fromkeys(candidates):
        split_paths = {split: _resolve_split(candidate, payload[split]) for split in REQUIRED_SPLITS}
        exists = all(path.exists() for values in split_paths.values() for path in values)
        checked.append({"root": str(candidate), "exists": exists, "splits": {k: [str(p) for p in v] for k, v in split_paths.items()}})
        if exists:
            names = payload["names"]
            nc = int(payload.get("nc", len(names)))
            if nc != len(names):
                raise ValueError(f"Dataset nc={nc} differs from len(names)={len(names)}")
            return payload, candidate, split_paths
    raise FileNotFoundError(f"No valid dataset root found. Checked: {json.dumps(checked, ensure_ascii=False)}")


def _images_from_entry(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    if path.is_file() and path.suffix.lower() == ".txt":
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    raise ValueError(f"Unsupported split entry: {path}")


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def audit_dataset(config: FormalConfig) -> dict[str, Any]:
    payload, root, split_paths = resolve_dataset(config)
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "source_yaml": str(config.drive_data_yaml),
        "resolved_root": str(root),
        "read_only": True,
        "names": payload["names"],
        "nc": int(payload.get("nc", len(payload["names"]))),
        "splits": {},
        "known_summary_comparison": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for split, entries in split_paths.items():
        images = sorted({image.resolve() for entry in entries for image in _images_from_entry(entry)})
        labels = [_label_for_image(image) for image in images]
        missing_labels: list[str] = []
        empty = 0
        instances = 0
        invalid_rows = 0
        classes: set[int] = set()
        for image, label in zip(images, labels, strict=True):
            if not label.is_file():
                missing_labels.append(str(label))
                continue
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty += 1
            for line in lines:
                fields = line.split()
                try:
                    cls = int(float(fields[0]))
                    coords = [float(value) for value in fields[1:]]
                    valid = len(coords) >= 4 and all(value == value for value in coords)
                except (ValueError, IndexError):
                    valid = False
                    cls = -1
                if valid:
                    instances += 1
                    classes.add(cls)
                else:
                    invalid_rows += 1
        split_report = {
            "images": len(images),
            "labels": sum(path.is_file() for path in labels),
            "empty_labels": empty,
            "instances": instances,
            "missing_labels": len(missing_labels),
            "missing_label_examples": missing_labels[:20],
            "invalid_label_rows": invalid_rows,
            "classes": sorted(classes),
        }
        report["splits"][split] = split_report
        expected = KNOWN_AUDITED_COUNTS.get(split, {})
        report["known_summary_comparison"][split] = {
            key: {"current": split_report[key], "previous_audit": value, "matches": split_report[key] == value}
            for key, value in expected.items()
        }
        csv_rows.append({"split": split, **{key: split_report[key] for key in ("images", "labels", "empty_labels", "instances", "missing_labels", "invalid_label_rows")}})
    if any(item["missing_labels"] or item["invalid_label_rows"] for item in report["splits"].values()):
        raise ValueError("Dataset audit found missing labels or invalid label rows; source was not modified.")
    audit_dir = config.protocol_staging_dir
    write_json(audit_dir / "dataset_runtime_audit.json", report)
    with (audit_dir / "dataset_runtime_audit.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    shutil.copyfile(config.drive_data_yaml, audit_dir / "data_original.yaml")
    return report


def _copy_one(pair: tuple[Path, Path]) -> tuple[int, bool]:
    source, destination = pair
    size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        return size, False
    shutil.copyfile(source, destination)
    return size, True


def copy_dataset_to_local(config: FormalConfig) -> Path:
    payload, source_root, _ = resolve_dataset(config)
    destination_root = Path(config.local_data_root)
    source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"No dataset files under {source_root}")
    jobs = [(source, destination_root / source.relative_to(source_root)) for source in source_files]
    total_bytes = sum(source.stat().st_size for source in source_files)
    copied_files = copied_bytes = 0
    with (
        tqdm(total=len(jobs), unit="file", desc="Dataset files") as files_bar,
        tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Dataset bytes") as bytes_bar,
        concurrent.futures.ThreadPoolExecutor(max_workers=config.copy_workers) as pool,
    ):
        for size, copied in pool.map(_copy_one, jobs, chunksize=8):
            files_bar.update(1)
            bytes_bar.update(size)
            copied_files += int(copied)
            copied_bytes += size if copied else 0
            files_bar.set_postfix(copied=copied_files, workers=config.copy_workers, refresh=False)
    missing = [str(destination) for _, destination in jobs if not destination.is_file()]
    mismatched = [str(destination) for source, destination in jobs if destination.is_file() and source.stat().st_size != destination.stat().st_size]
    report = {
        "generated_at": utc_now(),
        "source": str(source_root),
        "destination": str(destination_root),
        "source_files": len(source_files),
        "source_bytes": total_bytes,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "missing_after_copy": missing,
        "size_mismatches": mismatched,
        "workers": config.copy_workers,
    }
    if missing or mismatched:
        raise IOError(f"Dataset copy verification failed: {report}")
    protocol_dir = config.protocol_staging_dir
    write_json(protocol_dir / "dataset_copy_report.json", report)
    local = dict(payload)
    local["path"] = str(destination_root)
    config.local_yaml.parent.mkdir(parents=True, exist_ok=True)
    config.local_yaml.write_text(yaml.safe_dump(local, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.copyfile(config.local_yaml, protocol_dir / "data_runtime_local.yaml")
    return config.local_yaml


def register_modules() -> None:
    from custom_modules.register import register_custom_modules

    register_custom_modules()


def _model_with_nc(model_yaml: Path, nc: int):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    wrapper = YOLO(str(model_yaml), verbose=False)
    if int(wrapper.model.model[-1].nc) != nc:
        wrapper.model = DetectionModel(cfg=str(model_yaml), ch=3, nc=nc, verbose=False)
        wrapper.task = "detect"
        wrapper.ckpt = wrapper.ckpt or {}
    return wrapper


def _module_prefix(key: str) -> str:
    parts = key.split(".")
    return ".".join(parts[: min(len(parts), 4)])


def _direct_official_transfer(target, official_weights: str | Path) -> dict[str, Any]:
    from ultralytics import YOLO

    source = YOLO(str(official_weights), verbose=False)
    source_state = source.model.float().state_dict()
    target_state = target.model.state_dict()
    mapping = {key: key for key, value in target_state.items() if key in source_state and tuple(value.shape) == tuple(source_state[key].shape)}
    return _apply_mapping(target, source_state, mapping, official_weights, "same_name_same_shape")


def _shifted_official_transfer(target, official_weights: str | Path, nc: int) -> dict[str, Any]:
    """Map only official tensors while accounting for the VGUP layer-0 shift."""
    from ultralytics import YOLO

    source = YOLO(str(official_weights), verbose=False)
    source_state = source.model.float().state_dict()
    target_state = target.model.state_dict()
    mapping: dict[str, str] = {}
    for target_key, target_value in target_state.items():
        parts = target_key.split(".")
        if len(parts) < 3 or parts[0] != "model":
            continue
        try:
            layer = int(parts[1])
        except ValueError:
            continue
        if layer == 0:
            continue
        parts[1] = str(layer - 1)
        source_key = ".".join(parts)
        if source_key in source_state and tuple(source_state[source_key].shape) == tuple(target_value.shape):
            mapping[target_key] = source_key
    return _apply_mapping(target, source_state, mapping, official_weights, "official_same_shape_with_input_layer_shift")


def _apply_mapping(target, source_state: dict[str, torch.Tensor], mapping: dict[str, str], weights: str | Path, method: str) -> dict[str, Any]:
    target_state = target.model.state_dict()
    compatible = {target_key: source_state[source_key].detach().cpu() for target_key, source_key in mapping.items()}
    result = target.model.load_state_dict(compatible, strict=False)
    loaded = target.model.state_dict()
    verification_failures = [key for key, expected in compatible.items() if not torch.equal(loaded[key].detach().cpu(), expected)]
    parameter_names = set(dict(target.model.named_parameters()))
    total_parameter_elements = sum(value.numel() for value in target.model.parameters())
    inherited_parameter_elements = sum(target_state[key].numel() for key in compatible if key in parameter_names)
    unmatched = sorted(set(target_state) - set(compatible))
    return {
        "official_weights": str(weights),
        "method": method,
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
        "loaded_tensors": len(compatible),
        "tensor_inheritance_ratio": len(compatible) / len(target_state),
        "target_parameter_elements": total_parameter_elements,
        "loaded_parameter_elements": inherited_parameter_elements,
        "parameter_element_inheritance_ratio": inherited_parameter_elements / total_parameter_elements,
        "unmatched_target_keys": unmatched,
        "major_unmatched_modules": sorted({_module_prefix(key) for key in unmatched}),
        "missing_after_load": list(result.missing_keys),
        "unexpected_after_load": list(result.unexpected_keys),
        "verification_failures": verification_failures,
        "sample_mapping": dict(list(sorted(mapping.items()))[:30]),
        "passed": not result.unexpected_keys and not verification_failures and set(result.missing_keys).issubset(set(unmatched)),
    }


def build_and_initialize(config: FormalConfig, official_weights: str | Path = "yolo11n.pt"):
    register_modules()
    payload = yaml.safe_load(config.local_yaml.read_text(encoding="utf-8"))
    nc = int(payload.get("nc", len(payload["names"])))
    torch.manual_seed(config.seed)
    model = _model_with_nc(config.model_yaml, nc)
    if config.spec.get("shifted_official_mapping"):
        report = _shifted_official_transfer(model, official_weights, nc)
    else:
        report = _direct_official_transfer(model, official_weights)
    if not report["passed"]:
        raise RuntimeError(f"Official pretrained transfer failed: {report['verification_failures']}")
    output_dir = config.protocol_staging_dir
    write_json(output_dir / "pretrained_transfer_report.json", report)
    lines = [
        f"Experiment: {config.experiment_id}",
        f"Official weights: {official_weights}",
        f"Method: {report['method']}",
        f"Loaded/Total tensors: {report['loaded_tensors']}/{report['target_state_tensors']}",
        f"Tensor ratio: {report['tensor_inheritance_ratio']:.6%}",
        f"Loaded/Total parameter elements: {report['loaded_parameter_elements']}/{report['target_parameter_elements']}",
        f"Parameter ratio: {report['parameter_element_inheritance_ratio']:.6%}",
        "Major unmatched modules:",
        *[f"  - {item}" for item in report["major_unmatched_modules"]],
    ]
    (output_dir / "pretrained_transfer_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model, report


def _flatten_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _flatten_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten_tensors(child)


def audit_model(config: FormalConfig, model, *, backward_imgsz: int = 64) -> dict[str, Any]:
    layers = model.model.model
    detect = layers[-1]
    types = [type(layer).__name__ for layer in layers]
    inception = [index for index, name in enumerate(types) if name == "C3k2_InceptionDW"]
    dysample = [index for index, name in enumerate(types) if name == "DySample"]
    scam = [index for index, name in enumerate(types) if name == "SCAM"]
    ca_scam = [index for index, name in enumerate(types) if name == "CASCAM"]
    vgup = [index for index, name in enumerate(types) if name == "VGUPPreprocessor"]
    expected_inception = [] if config.experiment_id == "A0_yolo11n" else ([3, 5] if vgup else [2, 4])
    expected_scam = 3 if config.spec["specialty"] in {"scam", "vgup"} else 0
    expected_ca = 3 if config.spec["specialty"] == "ca_scam" else 0
    structure_checks = {
        "detect_from": list(detect.f) == config.spec["detect_from"],
        "detect_strides": [float(value) for value in model.model.stride] == config.spec["strides"],
        "inceptiondw_scope": inception == expected_inception,
        "dysample_count": len(dysample) == (2 if config.spec["specialty"] in {"dpls", "scam", "vgup", "ca_scam"} else 0),
        "scam_count": len(scam) == expected_scam,
        "ca_scam_count": len(ca_scam) == expected_ca,
        "vgup_count": len(vgup) == (1 if config.spec["specialty"] in {"vgup", "ca_scam"} else 0),
        "ca_scam_replaces_scam": not (scam and ca_scam),
    }
    network = model.model.cpu().train()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    image = torch.randn(1, 3, backward_imgsz, backward_imgsz, generator=generator, requires_grad=True)
    output = network(image)
    tensors = list(_flatten_tensors(output))
    loss = sum(value.float().square().mean() for value in tensors)
    loss.backward()
    gradients = [parameter.grad for parameter in network.parameters() if parameter.requires_grad and parameter.grad is not None]
    smoke_checks = {
        "outputs_exist": bool(tensors),
        "outputs_finite": bool(tensors) and all(torch.isfinite(value).all().item() for value in tensors),
        "input_gradient_finite": image.grad is not None and torch.isfinite(image.grad).all().item(),
        "parameter_gradients_finite": bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients),
    }
    report = {
        "experiment_id": config.experiment_id,
        "model_yaml": str(config.model_yaml),
        "layer_types": types,
        "inceptiondw_indices": inception,
        "dysample_indices": dysample,
        "scam_indices": scam,
        "ca_scam_indices": ca_scam,
        "vgup_indices": vgup,
        "detect_from": list(detect.f),
        "strides": [float(value) for value in model.model.stride],
        "structure_checks": structure_checks,
        "cpu_forward_backward_checks": smoke_checks,
        "passed": all(structure_checks.values()) and all(smoke_checks.values()),
    }
    write_json(config.protocol_staging_dir / "model_structure_audit.json", report)
    if not report["passed"]:
        raise AssertionError(f"Model audit failed: {report}")
    model.model.to(config.device if torch.cuda.is_available() else "cpu")
    return report


class _Tee(TextIO):
    def __init__(self, original: TextIO, log: TextIO) -> None:
        self.original, self.log = original, log

    def write(self, text: str) -> int:
        self.original.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self) -> None:
        self.original.flush()
        self.log.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


@contextlib.contextmanager
def tee_console(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_stdout, log), _Tee(old_stderr, log)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class AtomicDriveMirror:
    """Non-blocking Drive mirror using immutable local snapshots and atomic replace."""

    def __init__(self, local_root: Path, drive_root: Path) -> None:
        self.local_root = local_root
        self.drive_root = drive_root
        self.snapshot_root = local_root.parent / f".{local_root.name}_mirror_snapshots"
        self.jobs: queue.Queue[tuple[Path, Path] | None] = queue.Queue()
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._worker, name=f"mirror-{local_root.name}", daemon=True)
        self.drive_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def _worker(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                self.jobs.task_done()
                return
            snapshot, destination = job
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.name}.{time.time_ns()}.tmp")
                shutil.copyfile(snapshot, temporary)
                os.replace(temporary, destination)
            except Exception as error:  # mirror failure must be visible but not kill training
                message = f"Drive mirror failed for {destination}: {error}"
                self.errors.append(message)
                print(f"WARNING: {message}", file=sys.__stderr__)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    snapshot.unlink()
                self.jobs.task_done()

    def enqueue(self, relative: str | Path) -> None:
        source = self.local_root / relative
        if not source.is_file():
            return
        snapshot = self.snapshot_root / f"{time.time_ns()}_{source.name}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        self.jobs.put((snapshot, self.drive_root / relative))

    def enqueue_training_state(self) -> None:
        for relative in (
            "results.csv",
            "args.yaml",
            "train_console.log",
            "experiment_state.json",
            "weights/last.pt",
            "weights/best.pt",
        ):
            self.enqueue(relative)
        weights = self.local_root / "weights"
        if weights.is_dir():
            for path in sorted(weights.glob("epoch*.pt")):
                self.enqueue(path.relative_to(self.local_root))

    def sync_tree(self) -> None:
        for source in sorted(path for path in self.local_root.rglob("*") if path.is_file()):
            if self.snapshot_root in source.parents:
                continue
            self.enqueue(source.relative_to(self.local_root))

    def close(self) -> None:
        self.jobs.join()
        self.jobs.put(None)
        self.jobs.join()
        self.thread.join(timeout=30)
        if self.errors:
            print("WARNING: Drive mirror completed with errors:\n" + "\n".join(self.errors))


def capture_environment(config: FormalConfig, phase: str) -> dict[str, Any]:
    environment = require_environment()
    environment.update({"phase": phase, "captured_at": utc_now(), "git_commit": git_output("rev-parse", "HEAD")})
    output = config.protocol_staging_dir if phase == "start" else config.run_dir / "protocol"
    write_json(output / f"environment_{phase}.json", environment)
    if phase == "start":
        write_json(output / "environment.json", environment)
        (output / "pip_freeze.txt").write_text(
            subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout,
            encoding="utf-8",
        )
        gpu_text = "CUDA unavailable\n"
        if shutil.which("nvidia-smi"):
            gpu_text = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout
        (output / "gpu_info.txt").write_text(gpu_text, encoding="utf-8")
        (output / "git_commit.txt").write_text(environment["git_commit"] + "\n", encoding="utf-8")
        (output / "git_status.txt").write_text(git_output("status", "--short") + "\n", encoding="utf-8")
        (output / "git_diff.txt").write_text(git_output("diff", "--no-ext-diff") + "\n", encoding="utf-8")
        shutil.copyfile(config.model_yaml, output / "model.yaml")
    return environment


def restore_or_guard_run(config: FormalConfig) -> str:
    completed = config.drive_dir / "COMPLETED.ok"
    local_completed = config.run_dir / "COMPLETED.ok"
    if completed.is_file() or local_completed.is_file():
        raise FileExistsError(f"{config.experiment_id} is already complete; refusing to overwrite it.")
    local_last = config.run_dir / "weights" / "last.pt"
    drive_last = config.drive_dir / "weights" / "last.pt"
    if not local_last.is_file() and drive_last.is_file():
        if config.run_dir.exists() and any(config.run_dir.iterdir()):
            raise FileExistsError(f"Local run directory is non-empty but has no last.pt: {config.run_dir}")
        shutil.copytree(config.drive_dir, config.run_dir, dirs_exist_ok=True)
    if local_last.is_file():
        state_path = config.run_dir / "experiment_state.json"
        if not state_path.is_file():
            raise RuntimeError("Refusing resume: experiment_state.json is missing.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("experiment_id") != config.experiment_id:
            raise RuntimeError(f"Refusing cross-experiment resume: {state.get('experiment_id')} != {config.experiment_id}")
        return "resume"
    training_artifacts = []
    if config.run_dir.is_dir():
        allowed = {"protocol"}
        training_artifacts = [path.name for path in config.run_dir.iterdir() if path.name not in allowed]
    if training_artifacts:
        raise FileExistsError(f"Run directory has artifacts but no resumable last.pt: {training_artifacts}")
    return "new"


def _state(config: FormalConfig, status: str, **extra: Any) -> dict[str, Any]:
    payload = {"experiment_id": config.experiment_id, "status": status, "updated_at": utc_now(), **extra}
    write_json(config.run_dir / "experiment_state.json", payload)
    return payload


def train_foreground(config: FormalConfig, initialized_model=None):
    """Run the official Ultralytics API in this kernel with safe resume/mirroring."""
    mode = restore_or_guard_run(config)
    config.drive_dir.mkdir(parents=True, exist_ok=True)
    mirror = AtomicDriveMirror(config.run_dir, config.drive_dir)
    console_path = (
        config.run_dir / "train_console.log"
        if mode == "resume"
        else config.protocol_staging_dir / "train_console.log"
    )

    def initialize_run(trainer) -> None:
        actual = Path(trainer.save_dir).resolve()
        expected = config.run_dir.resolve()
        if actual != expected:
            raise RuntimeError(f"Ultralytics save_dir drifted from the fixed experiment directory: {actual} != {expected}")
        (config.run_dir / "protocol").mkdir(parents=True, exist_ok=True)
        shutil.copytree(config.protocol_staging_dir, config.run_dir / "protocol", dirs_exist_ok=True)
        if console_path.is_file() and console_path != config.run_dir / "train_console.log":
            shutil.copyfile(console_path, config.run_dir / "train_console.log")
        (config.run_dir / "RUNNING.lock").write_text(f"{utc_now()}\n", encoding="utf-8")
        _state(config, "running", mode=mode, target_epochs=config.epochs)
        mirror.enqueue_training_state()

    def sync_epoch(trainer) -> None:
        if console_path.is_file() and console_path != config.run_dir / "train_console.log":
            shutil.copyfile(console_path, config.run_dir / "train_console.log")
        _state(config, "running", mode=mode, epoch=int(trainer.epoch) + 1, target_epochs=config.epochs)
        mirror.enqueue_training_state()

    def sync_checkpoint(_trainer) -> None:
        mirror.enqueue_training_state()

    try:
        if mode == "resume":
            from ultralytics import YOLO

            model = YOLO(str(config.run_dir / "weights" / "last.pt"))
        else:
            if initialized_model is None:
                initialized_model, _ = build_and_initialize(config)
            model = initialized_model
        model.add_callback("on_pretrain_routine_start", initialize_run)
        model.add_callback("on_fit_epoch_end", sync_epoch)
        model.add_callback("on_model_save", sync_checkpoint)
        with tee_console(console_path):
            if mode == "resume":
                results = model.train(resume=True)
            else:
                results = model.train(
                    data=str(config.local_yaml),
                    epochs=config.epochs,
                    imgsz=config.imgsz,
                    batch=config.batch,
                    workers=config.workers,
                    seed=config.seed,
                    cache=config.cache,
                    deterministic=config.deterministic,
                    device=config.device,
                    plots=True,
                    save=True,
                    save_period=config.save_period,
                    project=config.local_runs_root,
                    name=config.experiment_id,
                    exist_ok=False,
                    pretrained=False,
                )
        if console_path.is_file() and console_path != config.run_dir / "train_console.log":
            shutil.copyfile(console_path, config.run_dir / "train_console.log")
        _state(config, "trained", mode=mode, target_epochs=config.epochs)
        mirror.enqueue_training_state()
        return model, results, mirror
    except Exception as error:
        failure = {
            "experiment_id": config.experiment_id,
            "failed_at": utc_now(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(config.run_dir / "FAILED.json", failure)
        mirror.enqueue("FAILED.json")
        mirror.close()
        raise


def best_epoch_summary(config: FormalConfig) -> dict[str, Any]:
    import pandas as pd

    path = config.run_dir / "results.csv"
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    metric = next(column for column in frame.columns if column.endswith("mAP50-95(B)"))
    index = int(frame[metric].astype(float).idxmax())
    row = frame.loc[index]
    lookup = {
        "precision": next(column for column in frame.columns if column.endswith("precision(B)")),
        "recall": next(column for column in frame.columns if column.endswith("recall(B)")),
        "map50": next(column for column in frame.columns if column.endswith("mAP50(B)")),
        "map50_95": metric,
    }
    summary = {
        "experiment_id": config.experiment_id,
        "selection_rule": "maximum validation mAP50-95",
        "best_epoch": int(row["epoch"]) + 1,
        **{key: float(row[column]) for key, column in lookup.items()},
        "losses": {column: float(row[column]) for column in frame.columns if "loss" in column},
        "training_time_seconds": float(frame["time"].iloc[-1]) if "time" in frame else None,
    }
    write_json(config.run_dir / "best_epoch_summary.json", summary)
    with (config.run_dir / "best_epoch_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def write_checksums(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    output = root / "artifact_checksums.sha256"
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != output):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output


def finalize_run(config: FormalConfig, model, mirror: AtomicDriveMirror | None = None) -> dict[str, Any]:
    """Generate post-training artifacts, mark complete only after all required steps."""
    from tools.paper_artifacts.generate_specialty_artifacts import generate_specialty_artifacts
    from tools.paper_artifacts.per_image_evaluation import evaluate_per_image
    from tools.paper_artifacts.performance_profile import profile_model
    from tools.paper_artifacts.export_experiment_bundle import export_bundle

    best = config.run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce {best}")
    from ultralytics import YOLO

    best_model = YOLO(str(best))
    validation = best_model.val(
        data=str(config.local_yaml),
        split="val",
        imgsz=config.imgsz,
        batch=config.batch,
        workers=config.workers,
        device=config.device,
        augment=False,
        plots=True,
        project=str(config.run_dir / "validation"),
        name="val",
        exist_ok=True,
    )
    validation_summary = {
        "precision": float(validation.box.mp),
        "recall": float(validation.box.mr),
        "map50": float(validation.box.map50),
        "map75": float(validation.box.map75),
        "map50_95": float(validation.box.map),
    }
    write_json(config.run_dir / "validation_metrics.json", validation_summary)
    if config.run_test_evaluation:
        test_metrics = best_model.val(
            data=str(config.local_yaml),
            split="test",
            imgsz=config.imgsz,
            batch=config.batch,
            workers=config.workers,
            device=config.device,
            augment=False,
            plots=False,
            project=str(config.run_dir / "validation"),
            name="test",
            exist_ok=True,
        )
        write_json(config.run_dir / "test_metrics.json", {
            "selection_prohibited": True,
            "precision": float(test_metrics.box.mp),
            "recall": float(test_metrics.box.mr),
            "map50": float(test_metrics.box.map50),
            "map75": float(test_metrics.box.map75),
            "map50_95": float(test_metrics.box.map),
        })
    predictions = evaluate_per_image(config, best_model)
    complexity = profile_model(config, best_model)
    specialty = generate_specialty_artifacts(config, best_model, predictions)
    summary = best_epoch_summary(config)
    environment_end = capture_environment(config, "end")
    protocol = config.run_dir / "protocol"
    manifest = {
        "experiment_id": config.experiment_id,
        "model_name": config.spec["model_name"],
        "git_commit": git_output("rev-parse", "HEAD"),
        "model_yaml_sha256": sha256_file(protocol / "model.yaml"),
        "data_yaml_sha256": sha256_file(protocol / "data_runtime_local.yaml"),
        "dataset_audit_sha256": sha256_file(protocol / "dataset_runtime_audit.json"),
        "seed": config.seed,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "workers": config.workers,
        "optimizer": "Ultralytics default/auto; see args.yaml",
        "learning_rate": "Ultralytics default/auto; see args.yaml",
        "start_time": json.loads((config.run_dir / "experiment_state.json").read_text(encoding="utf-8")).get("updated_at"),
        "end_time": utc_now(),
        "gpu": environment_end["gpu"],
        "python_version": environment_end["python"],
        "torch_version": environment_end["torch"],
        "cuda_version": environment_end["cuda"],
        "ultralytics_version": environment_end["ultralytics"],
        "best_epoch": summary["best_epoch"],
        "best_metrics": validation_summary,
        "params": complexity["parameters"],
        "gflops": complexity["gflops"],
        "specialty_artifacts": specialty,
        "per_image_rows": predictions["images"],
        "artifact_list": sorted(path.relative_to(config.run_dir).as_posix() for path in config.run_dir.rglob("*") if path.is_file()),
    }
    write_json(config.run_dir / "run_manifest.json", manifest)
    write_checksums(config.run_dir)
    bundle = export_bundle(config.run_dir, config.drive_dir / f"{config.experiment_id}_paper_artifacts.zip")
    (config.run_dir / "COMPLETED.ok").write_text(f"{utc_now()}\n{bundle}\n", encoding="utf-8")
    with contextlib.suppress(FileNotFoundError):
        (config.run_dir / "RUNNING.lock").unlink()
    _state(config, "completed", bundle=str(bundle))
    if mirror is None:
        mirror = AtomicDriveMirror(config.run_dir, config.drive_dir)
    mirror.sync_tree()
    mirror.close()
    return manifest


def prepare_experiment(config: FormalConfig):
    """Mounting/cloning happens in the notebook; this prepares all local inputs."""
    config.protocol_staging_dir.mkdir(parents=True, exist_ok=True)
    capture_environment(config, "start")
    audit = audit_dataset(config)
    local_yaml = copy_dataset_to_local(config)
    model, transfer = build_and_initialize(config)
    structure = audit_model(config, model)
    write_json(config.protocol_staging_dir / "formal_config.json", asdict(config))
    return {"audit": audit, "local_yaml": str(local_yaml), "model": model, "transfer": transfer, "structure": structure}


__all__ = [
    "EXPERIMENTS",
    "FormalConfig",
    "audit_dataset",
    "audit_model",
    "build_and_initialize",
    "copy_dataset_to_local",
    "finalize_run",
    "prepare_experiment",
    "train_foreground",
    "write_checksums",
]
