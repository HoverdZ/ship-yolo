"""Foreground-only, auditable execution protocol for formal experiments.

Training is always a direct call to the official ``YOLO.train`` API in the
current Colab kernel. This module never launches training in a subprocess.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from tools.formal_experiments.registry import (
    ROOT,
    load_registry,
    load_yaml,
    resolve_run,
)
from tools.paper_artifacts.formal_protocol import (
    AtomicDriveMirror,
    audit_dataset,
    copy_dataset_to_local,
    sha256_file,
    write_checksums,
    write_json,
)

ULTRALYTICS_VERSION = "8.4.92"
EXPECTED_RUN_ARTIFACTS = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "results.png",
    "args.yaml",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "environment.txt",
    "pip_freeze.txt",
    "git_commit.txt",
    "model_summary.txt",
    "model_info.json",
    "run_manifest.json",
    "artifact_checksums.sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


@dataclass(frozen=True)
class FormalRunConfig:
    """Resolved registry and protocol settings for one seed instance."""

    experiment_id: str
    run_id: str
    seed: int
    model_yaml: Path
    initialization_weight: str
    expected_detect_strides: tuple[float, float, float]
    spec: dict[str, Any]
    training: dict[str, Any]
    drive_project_root: str
    drive_data_yaml: str
    drive_data_root: str | None
    local_data_root: str
    local_yaml: Path
    local_runs_root: str
    copy_workers: int
    device: int | str
    run_training: bool = False
    run_test_evaluation: bool = False
    enforce_environment_lock: bool = True

    @classmethod
    def from_registry(
        cls,
        run_or_alias: str,
        *,
        seed: int | None = None,
        run_training: bool = False,
        run_test_evaluation: bool = False,
        data_yaml_override: str | None = None,
        drive_data_root_override: str | None = None,
    ) -> "FormalRunConfig":
        registry = load_registry()
        run_id, spec = resolve_run(run_or_alias, registry)
        protocol = load_yaml(ROOT / registry["training_config"])
        training = dict(protocol["training"])
        selected_seed = int(spec["seed"] if seed is None else seed)
        if selected_seed not in training["seeds"]:
            raise ValueError(
                f"seed={selected_seed} is outside configured seeds "
                f"{training['seeds']}"
            )
        dataset = protocol["dataset"]
        outputs = protocol["outputs"]
        data_yaml = data_yaml_override or spec["data_yaml"]
        if spec["dataset_id"] == "external_dataset_pending" and not data_yaml_override:
            raise RuntimeError(
                f"{run_id} is blocked until a second dataset is selected. "
                "Pass data_yaml_override after completing the integration audit."
            )
        return cls(
            experiment_id=run_id,
            run_id=run_id,
            seed=selected_seed,
            model_yaml=ROOT / spec["model_yaml"],
            initialization_weight=spec["initialization_weight"],
            expected_detect_strides=tuple(
                float(value) for value in spec["expected_detect_strides"]
            ),
            spec=dict(spec),
            training=training,
            drive_project_root=outputs["drive_project_root"],
            drive_data_yaml=data_yaml,
            drive_data_root=(
                drive_data_root_override
                if drive_data_root_override is not None
                else dataset.get("drive_root")
            ),
            local_data_root=dataset["local_root"],
            local_yaml=Path(dataset["local_data_yaml"]),
            local_runs_root=outputs["local_root"],
            copy_workers=int(dataset["copy_workers"]),
            device=training["device"],
            run_training=bool(run_training),
            run_test_evaluation=bool(run_test_evaluation),
            enforce_environment_lock=bool(
                protocol["environment"]["enforce_shared_environment_lock"]
            ),
        )

    @property
    def run_name(self) -> str:
        return f"seed_{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.local_runs_root) / self.run_id / self.run_name

    @property
    def drive_dir(self) -> Path:
        return (
            Path(self.drive_project_root)
            / "formal_experiments"
            / self.run_id
            / self.run_name
        )

    @property
    def protocol_staging_dir(self) -> Path:
        return (
            Path(self.local_runs_root)
            / ".protocol_staging"
            / self.run_id
            / self.run_name
        )

    @property
    def imgsz(self) -> int:
        return int(self.training["imgsz"])

    @property
    def epochs(self) -> int:
        return int(self.training["epochs"])

    @property
    def batch(self) -> int:
        return int(self.training["batch"])

    @property
    def workers(self) -> int:
        return int(self.training["workers"])

    @property
    def conf(self) -> float:
        return 0.25

    @property
    def iou(self) -> float:
        return 0.70

    @property
    def tiny_short_side(self) -> float:
        return 16.0

    @property
    def small_short_side(self) -> float:
        return 32.0


def environment_record() -> dict[str, Any]:
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected ultralytics=={ULTRALYTICS_VERSION}, "
            f"found {ultralytics.__version__}"
        )
    cuda = torch.version.cuda
    gpu = None
    memory = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory
    return {
        "captured_at": utc_now(),
        "python": platform.python_version(),
        "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
        "torch": torch.__version__,
        "cuda": cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": ultralytics.__version__,
        "gpu": gpu,
        "gpu_memory_bytes": memory,
        "platform": platform.platform(),
        "git_commit": git_output("rev-parse", "HEAD"),
    }


def enforce_environment_lock(
    config: FormalRunConfig,
    environment: dict[str, Any],
) -> Path:
    lock_path = (
        Path(config.drive_project_root)
        / "paper_artifacts"
        / "manifests"
        / "environment_lock.json"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    comparable = (
        "python_major_minor",
        "torch",
        "cuda",
        "ultralytics",
        "gpu",
    )
    if lock_path.is_file():
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"locked": locked.get(key), "current": environment.get(key)}
            for key in comparable
            if locked.get(key) != environment.get(key)
        }
        if mismatches and config.enforce_environment_lock:
            raise RuntimeError(
                "Colab environment differs from the formal environment lock: "
                + json.dumps(mismatches, ensure_ascii=False)
            )
    else:
        temporary = lock_path.with_name(f".{lock_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, lock_path)
    return lock_path


def capture_environment(
    config: FormalRunConfig,
    *,
    enforce_lock: bool = True,
) -> dict[str, Any]:
    environment = environment_record()
    staging = config.protocol_staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    if enforce_lock:
        enforce_environment_lock(config, environment)
    write_json(staging / "environment.json", environment)
    environment_text = "\n".join(
        f"{key}: {value}" for key, value in environment.items()
    )
    (staging / "environment.txt").write_text(
        environment_text + "\n",
        encoding="utf-8",
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (staging / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    (staging / "git_commit.txt").write_text(
        environment["git_commit"] + "\n",
        encoding="utf-8",
    )
    (staging / "git_status.txt").write_text(
        git_output("status", "--short") + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(config.model_yaml, staging / "model.yaml")
    write_json(staging / "resolved_run_config.json", _jsonable(asdict(config)))
    return environment


def snapshot_repository(config: FormalRunConfig) -> Path:
    """Write a tracked Git snapshot to Drive without copying credentials."""

    commit = git_output("rev-parse", "HEAD")
    output = (
        Path(config.drive_project_root)
        / "repository_snapshots"
        / f"ship-yolo_{commit[:12]}.zip"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file():
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "archive",
                "--format=zip",
                f"--output={output}",
                commit,
            ],
            check=True,
        )
    return output


def prepare_dataset(config: FormalRunConfig) -> tuple[Path, dict[str, Any]]:
    """Copy Drive data with realtime file/byte progress, then audit locally."""

    local_yaml = copy_dataset_to_local(config)
    local_config = replace(
        config,
        drive_data_yaml=str(local_yaml),
        drive_data_root=str(Path(config.local_data_root).resolve()),
    )
    print("Auditing the verified local copy (source Drive data is read-only)...")
    report = audit_dataset(local_config)
    report["audit_execution_root"] = "verified_local_copy"
    report["source_dataset_was_modified"] = False
    write_json(config.protocol_staging_dir / "dataset_runtime_audit.json", report)
    return local_yaml, report


def _model_with_nc(model_yaml: Path, nc: int):
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
        wrapper.ckpt = wrapper.ckpt or {}
    return wrapper


def _layer_mapping(run_id: str) -> dict[int, int]:
    if run_id in {"R00", "S00"}:
        return {index: index for index in range(24)}
    if run_id == "R11":
        return {index: index for index in range(23)}
    if run_id == "R12":
        return {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            11: 15,
            18: 16,
            20: 18,
        }
    input_shift = run_id in {
        "R06",
        "R07",
        "R08",
        "R09",
        "R10",
        "S01",
        "PLS_CA_SCAM_VGUP_150ep",
        "PLS_CA_SCAM_ERUP_150ep",
    }
    offset = 1 if input_shift else 0
    mapping = {index + offset: index for index in range(7)}
    mapping.update(
        {
            15 + offset: 17,
            17 + offset: 19,
            18 + offset: 20,
        }
    )
    return mapping


def _detect_mapping(run_id: str) -> tuple[int, int, dict[int, int]]:
    if run_id in {"R00", "S00"}:
        return 23, 23, {0: 0, 1: 1, 2: 2}
    if run_id == "R11":
        return 22, 22, {0: 0, 1: 1, 2: 2}
    if run_id == "R12":
        return 24, 22, {1: 0, 2: 1}
    if run_id in {"R01", "R02"}:
        target = 21
    elif run_id in {
        "R03",
        "R04",
        "R05A",
        "R05B",
        "PLS_SCAM_150ep",
        "PLS_CA_SCAM_150ep",
    }:
        target = 24
    else:
        target = 25
    # Target P2/P3/P4 channels are compatible with official P3/P4 heads.
    return target, 23, {0: 0, 1: 1, 2: 1}


def _map_layer_keys(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    layer_map: dict[int, int],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for target_key, target_value in target_state.items():
        parts = target_key.split(".")
        if len(parts) < 3 or parts[0] != "model":
            continue
        try:
            target_layer = int(parts[1])
        except ValueError:
            continue
        if target_layer not in layer_map:
            continue
        parts[1] = str(layer_map[target_layer])
        source_key = ".".join(parts)
        if (
            source_key in source_state
            and tuple(source_state[source_key].shape)
            == tuple(target_value.shape)
        ):
            mapping[target_key] = source_key
    return mapping


def _map_detect_keys(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    target_layer: int,
    source_layer: int,
    branch_map: dict[int, int],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    prefix = f"model.{target_layer}."
    for target_key, target_value in target_state.items():
        if not target_key.startswith(prefix):
            continue
        suffix = target_key[len(prefix):]
        parts = suffix.split(".")
        source_suffix = suffix
        if len(parts) >= 2 and parts[0] in {"cv2", "cv3"}:
            try:
                target_branch = int(parts[1])
            except ValueError:
                continue
            if target_branch not in branch_map:
                continue
            parts[1] = str(branch_map[target_branch])
            source_suffix = ".".join(parts)
        elif parts[0] != "dfl":
            continue
        source_key = f"model.{source_layer}.{source_suffix}"
        if (
            source_key in source_state
            and tuple(source_state[source_key].shape)
            == tuple(target_value.shape)
        ):
            mapping[target_key] = source_key
    return mapping


def _module_prefix(key: str) -> str:
    parts = key.split(".")
    return ".".join(parts[: min(4, len(parts))])


def transfer_official_weights(
    config: FormalRunConfig,
    target,
) -> dict[str, Any]:
    from ultralytics import YOLO

    source = YOLO(config.initialization_weight, verbose=False)
    source_state = source.model.float().state_dict()
    target_state = target.model.state_dict()
    layer_map = _layer_mapping(config.run_id)
    mapping = _map_layer_keys(target_state, source_state, layer_map)
    target_detect, source_detect, branch_map = _detect_mapping(config.run_id)
    mapping.update(
        _map_detect_keys(
            target_state,
            source_state,
            target_detect,
            source_detect,
            branch_map,
        )
    )
    compatible = {
        target_key: source_state[source_key].detach().cpu()
        for target_key, source_key in mapping.items()
    }
    result = target.model.load_state_dict(compatible, strict=False)
    loaded_state = target.model.state_dict()
    failures = [
        key
        for key, expected in compatible.items()
        if not torch.equal(loaded_state[key].detach().cpu(), expected)
    ]
    parameter_names = set(dict(target.model.named_parameters()))
    target_parameter_elements = sum(
        value.numel() for value in target.model.parameters()
    )
    loaded_parameter_elements = sum(
        target_state[key].numel()
        for key in compatible
        if key in parameter_names
    )
    unmatched = sorted(set(target_state) - set(compatible))
    report = {
        "official_weights": config.initialization_weight,
        "policy": "explicit_semantic_layer_and_same_shape_mapping",
        "layer_mapping": layer_map,
        "detect_branch_mapping": branch_map,
        "source_detect_layer": source_detect,
        "target_detect_layer": target_detect,
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
        "loaded_tensors": len(compatible),
        "loaded_total": f"{len(compatible)}/{len(target_state)}",
        "tensor_inheritance_ratio": len(compatible) / len(target_state),
        "target_parameter_elements": target_parameter_elements,
        "loaded_parameter_elements": loaded_parameter_elements,
        "parameter_element_inheritance_ratio": (
            loaded_parameter_elements / target_parameter_elements
        ),
        "loaded_target_keys": sorted(compatible),
        "loaded_source_mapping": dict(sorted(mapping.items())),
        "unmatched_target_keys": unmatched,
        "major_unmatched_modules": sorted(
            {_module_prefix(key) for key in unmatched}
        ),
        "missing_after_load": list(result.missing_keys),
        "unexpected_after_load": list(result.unexpected_keys),
        "verification_failures": failures,
        "passed": not result.unexpected_keys and not failures,
    }
    if not report["passed"]:
        raise RuntimeError(
            "Official pretrained transfer failed: "
            + json.dumps(report["verification_failures"][:20])
        )
    return report


def build_and_initialize(
    config: FormalRunConfig,
) -> tuple[Any, dict[str, Any]]:
    from custom_modules.register import register_custom_modules

    register_custom_modules()
    data = yaml.safe_load(config.local_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    nc = int(data.get("nc", len(names)))
    torch.manual_seed(config.seed)
    model = _model_with_nc(config.model_yaml, nc)
    report = transfer_official_weights(config, model)
    # Ultralytics 8.4.92 uses this truthy checkpoint marker to hand the
    # initialized in-memory model to DetectionTrainer. A callback verifies
    # every tensor before epoch 1, so no silent yolo*.pt override is possible.
    model.ckpt = {
        "model": model.model,
        "epoch": -1,
        "optimizer": None,
        "formal_pretrained_transfer": True,
    }
    staging = config.protocol_staging_dir
    write_json(staging / "pretrained_transfer_report.json", report)
    (staging / "pretrained_transfer_report.txt").write_text(
        "\n".join(
            (
                f"Run: {config.run_id}",
                f"Official weights: {config.initialization_weight}",
                f"Policy: {report['policy']}",
                f"Loaded/Total tensors: {report['loaded_total']}",
                "No preceding ablation checkpoint was used.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
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


def audit_model(
    config: FormalRunConfig,
    model,
    *,
    backward_imgsz: int = 64,
) -> dict[str, Any]:
    network = model.model.cpu()
    layers = network.model
    detect = layers[-1]
    types = [type(layer).__name__ for layer in layers]
    flags = config.spec["module_flags"]
    expected_counts = {
        "DySample": 2 if flags.get("dpls") else 0,
        "SCAM": 3 if flags.get("scam") else 0,
        "CASCAM": 3 if flags.get("ca_scam") == "bounded" else 0,
        "CASCAMFixedBeta": (
            3 if flags.get("ca_scam") == "fixed_beta" else 0
        ),
        "CASCAMUnbounded": (
            3 if flags.get("ca_scam") == "unbounded_beta" else 0
        ),
        "ERUPPreprocessor": 1 if flags.get("erup") else 0,
        "VGUPPreprocessor": 1 if flags.get("vgup") else 0,
    }
    structure_checks = {
        "detect_strides": [float(value) for value in network.stride]
        == list(config.expected_detect_strides),
        **{
            f"{name}_count": types.count(name) == count
            for name, count in expected_counts.items()
        },
    }
    vgup_details: dict[str, Any] | None = None
    if flags.get("vgup"):
        preprocessor = layers[0]
        vgup_details = {
            "use_global_gate": preprocessor.use_global_gate,
            "use_spatial_gate": preprocessor.use_spatial_gate,
            "expected_global_gate": bool(flags["global_gate"]),
            "expected_spatial_gate": bool(flags["spatial_gate"]),
        }
        structure_checks["vgup_global_gate"] = (
            preprocessor.use_global_gate == bool(flags["global_gate"])
        )
        structure_checks["vgup_spatial_gate"] = (
            preprocessor.use_spatial_gate == bool(flags["spatial_gate"])
        )
    original_training = network.training
    network.eval()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    image = torch.randn(
        1,
        3,
        backward_imgsz,
        backward_imgsz,
        generator=generator,
        requires_grad=True,
    )
    output = network(image)
    tensors = list(_flatten_tensors(output))
    loss = sum(value.float().square().mean() for value in tensors)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in network.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    smoke_checks = {
        "outputs_exist": bool(tensors),
        "outputs_finite": bool(tensors)
        and all(torch.isfinite(value).all().item() for value in tensors),
        "input_gradient_finite": image.grad is not None
        and torch.isfinite(image.grad).all().item(),
        "parameter_gradients_finite": bool(gradients)
        and all(torch.isfinite(value).all().item() for value in gradients),
    }
    report = {
        "experiment_id": config.experiment_id,
        "model_yaml": str(config.model_yaml),
        "layer_types": types,
        "detect_from": list(detect.f),
        "strides": [float(value) for value in network.stride],
        "parameter_count": sum(
            parameter.numel() for parameter in network.parameters()
        ),
        "module_counts": {
            name: types.count(name) for name in expected_counts
        },
        "vgup": vgup_details,
        "structure_checks": structure_checks,
        "cpu_forward_backward_checks": smoke_checks,
        "passed": all(structure_checks.values())
        and all(smoke_checks.values()),
    }
    write_json(config.protocol_staging_dir / "model_structure_audit.json", report)
    if not report["passed"]:
        raise AssertionError(f"Model audit failed: {report}")
    network.zero_grad(set_to_none=True)
    network.train(original_training)
    model.model.to(config.device if torch.cuda.is_available() else "cpu")
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _model_info(config: FormalRunConfig, model) -> dict[str, Any]:
    from ultralytics.utils.torch_utils import get_flops

    network = model.model
    info = {
        "run_id": config.run_id,
        "model_yaml": str(config.model_yaml),
        "parameters": sum(value.numel() for value in network.parameters()),
        "trainable_parameters": sum(
            value.numel() for value in network.parameters()
            if value.requires_grad
        ),
        "state_tensors": len(network.state_dict()),
        "layers": len(network.model),
        "gflops_imgsz_640": float(get_flops(network, imgsz=config.imgsz)),
        "detect_strides": [float(value) for value in network.stride],
    }
    write_json(config.protocol_staging_dir / "model_info.json", info)
    (config.protocol_staging_dir / "model_summary.txt").write_text(
        str(network) + "\n\n" + json.dumps(info, indent=2) + "\n",
        encoding="utf-8",
    )
    return info


def prepare_experiment(
    config: FormalRunConfig,
) -> dict[str, Any]:
    """Resolve environment/data/model without starting training."""

    config.protocol_staging_dir.mkdir(parents=True, exist_ok=True)
    environment = capture_environment(config)
    snapshot = snapshot_repository(config)
    local_yaml, dataset_audit = prepare_dataset(config)
    model, transfer = build_and_initialize(config)
    structure = audit_model(config, model)
    model_info = _model_info(config, model)
    return {
        "environment": environment,
        "repository_snapshot": str(snapshot),
        "local_yaml": str(local_yaml),
        "dataset_audit": dataset_audit,
        "model": model,
        "transfer": transfer,
        "structure": structure,
        "model_info": model_info,
    }


def _state_hash(state: dict[str, torch.Tensor]) -> str:
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


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _verify_trainer_handoff(
    config: FormalRunConfig,
    trainer_model,
    expected_state: dict[str, torch.Tensor],
    inherited_keys: list[str],
) -> dict[str, Any]:
    actual = _unwrap_model(trainer_model).state_dict()
    missing = sorted(set(expected_state) - set(actual))
    unexpected = sorted(set(actual) - set(expected_state))
    shape_mismatches = sorted(
        key
        for key in set(expected_state).intersection(actual)
        if tuple(expected_state[key].shape) != tuple(actual[key].shape)
    )
    value_mismatches = sorted(
        key
        for key in set(expected_state).intersection(actual)
        if key not in shape_mismatches
        and not torch.equal(expected_state[key], actual[key].detach().cpu())
    )
    all_mismatches = (
        set(missing) | set(unexpected) | set(shape_mismatches)
        | set(value_mismatches)
    )
    official_mismatches = sorted(set(inherited_keys).intersection(all_mismatches))
    report = {
        "expected_tensors": len(expected_state),
        "actual_tensors": len(actual),
        "exact_tensors": (
            len(expected_state)
            - len(missing)
            - len(shape_mismatches)
            - len(value_mismatches)
        ),
        "official_pretrained_tensors_expected": len(inherited_keys),
        "official_pretrained_tensors_preserved": (
            len(inherited_keys) - len(official_mismatches)
        ),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "value_mismatches": value_mismatches,
        "official_pretrained_mismatches": official_mismatches,
        "expected_state_sha256": _state_hash(expected_state),
        "actual_state_sha256": _state_hash(actual),
    }
    report["passed"] = not any(
        (missing, unexpected, shape_mismatches, value_mismatches)
    )
    write_json(config.run_dir / "trainer_handoff_report.json", report)
    if not report["passed"]:
        raise RuntimeError(
            "Trainer changed initialized tensors before epoch 1: "
            f"official_mismatches={len(official_mismatches)}, "
            f"all_value_mismatches={len(value_mismatches)}"
        )
    print(
        "Trainer handoff exact tensors:",
        f"{report['exact_tensors']}/{report['expected_tensors']}",
    )
    print(
        "Official pretrained tensors preserved:",
        f"{report['official_pretrained_tensors_preserved']}/"
        f"{report['official_pretrained_tensors_expected']}",
    )
    return report


def resolve_run_state(config: FormalRunConfig) -> str:
    """Return ``new`` or ``resume`` without overwriting a prior attempt."""

    completed = config.drive_dir / "COMPLETED.ok"
    if completed.is_file() or (config.run_dir / "COMPLETED.ok").is_file():
        raise FileExistsError(
            f"{config.run_id} seed {config.seed} is already complete."
        )
    local_last = config.run_dir / "weights" / "last.pt"
    drive_last = config.drive_dir / "weights" / "last.pt"
    if not local_last.is_file() and drive_last.is_file():
        if config.run_dir.is_dir() and any(config.run_dir.iterdir()):
            raise FileExistsError(
                f"Local run is non-empty but has no last.pt: {config.run_dir}"
            )
        shutil.copytree(config.drive_dir, config.run_dir, dirs_exist_ok=True)
    if (config.run_dir / "weights" / "last.pt").is_file():
        state_file = config.run_dir / "experiment_state.json"
        if not state_file.is_file():
            raise RuntimeError("Resume blocked: experiment_state.json is missing.")
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if (
            state.get("run_id") != config.run_id
            or int(state.get("seed", -1)) != config.seed
        ):
            raise RuntimeError(
                "Resume blocked: checkpoint directory belongs to another run."
            )
        return "resume"
    if config.run_dir.is_dir():
        allowed = {"protocol"}
        artifacts = [
            path.name
            for path in config.run_dir.iterdir()
            if path.name not in allowed
        ]
        if artifacts:
            raise FileExistsError(
                "Run directory contains non-resumable artifacts: "
                + ", ".join(artifacts)
            )
    return "new"


def _write_state(
    config: FormalRunConfig,
    status: str,
    **extra: Any,
) -> None:
    write_json(
        config.run_dir / "experiment_state.json",
        {
            "run_id": config.run_id,
            "seed": config.seed,
            "status": status,
            "updated_at": utc_now(),
            **extra,
        },
    )


def _copy_staging_to_run(config: FormalRunConfig) -> None:
    protocol = config.run_dir / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.protocol_staging_dir, protocol, dirs_exist_ok=True)
    for filename in (
        "environment.txt",
        "pip_freeze.txt",
        "git_commit.txt",
        "model_summary.txt",
        "model_info.json",
    ):
        source = config.protocol_staging_dir / filename
        if source.is_file():
            shutil.copyfile(source, config.run_dir / filename)


def train_foreground(
    config: FormalRunConfig,
    initialized_model=None,
):
    """Run official Ultralytics training directly in the current kernel."""

    mode = resolve_run_state(config)
    config.drive_dir.mkdir(parents=True, exist_ok=True)
    mirror = AtomicDriveMirror(config.run_dir, config.drive_dir)
    if mode == "resume":
        from ultralytics import YOLO

        model = YOLO(str(config.run_dir / "weights" / "last.pt"))
        inherited_keys: list[str] = []
    else:
        if initialized_model is None:
            initialized_model, transfer = build_and_initialize(config)
        else:
            transfer_path = (
                config.protocol_staging_dir
                / "pretrained_transfer_report.json"
            )
            transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
        model = initialized_model
        inherited_keys = list(transfer["loaded_target_keys"])
    expected_state = {
        key: value.detach().cpu().clone()
        for key, value in model.model.state_dict().items()
    }

    def on_start(trainer) -> None:
        actual = Path(trainer.save_dir).resolve()
        expected = config.run_dir.resolve()
        if actual != expected:
            raise RuntimeError(
                f"Ultralytics save_dir drift: {actual} != {expected}"
            )
        _copy_staging_to_run(config)
        (config.run_dir / "RUNNING.lock").write_text(
            utc_now() + "\n",
            encoding="utf-8",
        )
        _write_state(
            config,
            "running",
            mode=mode,
            target_epochs=config.epochs,
        )
        mirror.enqueue_training_state()

    def on_ready(trainer) -> None:
        _verify_trainer_handoff(
            config,
            trainer.model,
            expected_state,
            inherited_keys,
        )
        mirror.enqueue("trainer_handoff_report.json")

    def on_epoch(trainer) -> None:
        _write_state(
            config,
            "running",
            mode=mode,
            epoch=int(trainer.epoch) + 1,
            target_epochs=config.epochs,
        )
        mirror.enqueue_training_state()

    model.add_callback("on_pretrain_routine_start", on_start)
    model.add_callback("on_pretrain_routine_end", on_ready)
    model.add_callback("on_fit_epoch_end", on_epoch)
    model.add_callback(
        "on_model_save",
        lambda _trainer: mirror.enqueue_training_state(),
    )
    try:
        if mode == "resume":
            # Direct foreground call; official epoch output remains visible.
            results = model.train(resume=True)
        else:
            t = config.training
            # Direct foreground call; do not wrap, capture, or subprocess this.
            results = model.train(
                data=str(config.local_yaml),
                epochs=config.epochs,
                imgsz=config.imgsz,
                batch=config.batch,
                workers=config.workers,
                optimizer=t["optimizer"],
                lr0=t["lr0"],
                lrf=t["lrf"],
                momentum=t["momentum"],
                weight_decay=t["weight_decay"],
                warmup_epochs=t["warmup_epochs"],
                patience=t["patience"],
                seed=config.seed,
                cache=t["cache"],
                deterministic=t["deterministic"],
                device=config.device,
                mosaic=t["mosaic"],
                close_mosaic=t["close_mosaic"],
                scale=t["scale"],
                translate=t["translate"],
                box=t["box"],
                cls=t["cls"],
                dfl=t["dfl"],
                plots=t["plots"],
                save=True,
                save_period=t["save_period"],
                project=str(Path(config.local_runs_root) / config.run_id),
                name=config.run_name,
                exist_ok=False,
                # In Ultralytics 8.4.92 this passes the already initialized
                # in-memory model into DetectionTrainer. The callback above
                # asserts every tensor before epoch 1.
                pretrained=True,
            )
        _write_state(config, "trained", mode=mode)
        mirror.enqueue_training_state()
        return model, results, mirror
    except Exception as error:
        write_json(
            config.run_dir / "FAILED.json",
            {
                "run_id": config.run_id,
                "seed": config.seed,
                "failed_at": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        mirror.enqueue("FAILED.json")
        mirror.close()
        raise


def _best_epoch(config: FormalRunConfig) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(config.run_dir / "results.csv")
    frame.columns = [column.strip() for column in frame.columns]
    metric = next(
        column for column in frame if column.endswith("mAP50-95(B)")
    )
    index = int(frame[metric].astype(float).idxmax())
    row = frame.loc[index]

    def column(suffix: str) -> str:
        return next(name for name in frame if name.endswith(suffix))

    return {
        "selection_rule": "maximum validation mAP50-95",
        "best_epoch": int(row["epoch"]) + 1,
        "precision": float(row[column("precision(B)")]),
        "recall": float(row[column("recall(B)")]),
        "map50": float(row[column("mAP50(B)")]),
        "map50_95": float(row[metric]),
    }


def _artifact_inventory(run_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    present = sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    missing = []
    for relative in EXPECTED_RUN_ARTIFACTS:
        if relative not in present:
            reason = (
                "not generated by Ultralytics for this run/configuration"
                if relative.endswith((".jpg", ".png"))
                else "post-processing step did not produce this artifact"
            )
            missing.append({"path": relative, "reason": reason})
    return present, missing


def _export_zip(run_dir: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(
                item for item in run_dir.rglob("*") if item.is_file()
            ):
                archive.write(
                    path,
                    arcname=(
                        f"{run_dir.parent.name}/{run_dir.name}/"
                        f"{path.relative_to(run_dir).as_posix()}"
                    ),
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _seal_run(
    config: FormalRunConfig,
    manifest: dict[str, Any],
    mirror: AtomicDriveMirror | None = None,
) -> Path:
    """Seal stable final state before checksums, ZIP export, and Drive sync."""

    from tools.windows_collection import verify_checksum_manifest

    zip_path = (
        Path(config.drive_project_root)
        / "exports"
        / f"{config.run_id}_{config.run_name}.zip"
    )
    completed = config.run_dir / "COMPLETED.ok"
    drive_completed = config.drive_dir / "COMPLETED.ok"
    try:
        # Runtime markers and state must reach their final values before the
        # checksum manifest and ZIP are generated.
        with contextlib.suppress(FileNotFoundError):
            (config.run_dir / "RUNNING.lock").unlink()
        _write_state(config, "completed", export_zip=str(zip_path))
        completed.write_text(
            f"{utc_now()}\n{zip_path}\n",
            encoding="utf-8",
        )

        write_json(config.run_dir / "run_manifest.json", manifest)
        write_checksums(config.run_dir)
        present, missing = _artifact_inventory(config.run_dir)
        manifest["artifacts_present"] = present
        manifest["artifacts_missing"] = missing
        write_json(config.run_dir / "run_manifest.json", manifest)
        checksum_file = write_checksums(config.run_dir)

        failures = [
            row
            for row in verify_checksum_manifest(checksum_file)
            if not row["passed"]
        ]
        if failures:
            raise RuntimeError(
                "Final checksum verification failed: "
                + json.dumps(failures[:10], ensure_ascii=False)
            )

        _export_zip(config.run_dir, zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            damaged = archive.testzip()
        if damaged is not None:
            raise RuntimeError(f"Damaged file in final ZIP: {damaged}")

        if mirror is None:
            mirror = AtomicDriveMirror(config.run_dir, config.drive_dir)
        mirror.sync_tree()
        mirror.close()

        # AtomicDriveMirror copies current files but does not delete an older
        # Drive-side lock, so remove it after all queued jobs have finished.
        with contextlib.suppress(OSError):
            (config.drive_dir / "RUNNING.lock").unlink()
        return zip_path
    except Exception:
        completed.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            drive_completed.unlink()
        with contextlib.suppress(Exception):
            _write_state(
                config,
                "finalization_failed",
                export_zip=str(zip_path),
            )
        raise


def finalize_run(
    config: FormalRunConfig,
    mirror: AtomicDriveMirror | None = None,
) -> dict[str, Any]:
    """Validate best.pt, write manifest/checksums, mirror, and ZIP."""

    from custom_modules.register import register_custom_modules
    from tools.paper_artifacts.per_image_evaluation import evaluate_per_image
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    register_custom_modules()
    best = config.run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    model = YOLO(str(best))
    metrics = model.val(
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
    validation = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95": float(metrics.box.map),
    }
    write_json(config.run_dir / "validation_metrics.json", validation)
    per_image = evaluate_per_image(config, model)
    if config.run_test_evaluation:
        test = model.val(
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
        write_json(
            config.run_dir / "test_metrics.json",
            {
                "selection_prohibited": True,
                "precision": float(test.box.mp),
                "recall": float(test.box.mr),
                "map50": float(test.box.map50),
                "map75": float(test.box.map75),
                "map50_95": float(test.box.map),
            },
        )
    network = model.model
    complexity = {
        "parameters": sum(value.numel() for value in network.parameters()),
        "trainable_parameters": sum(
            value.numel()
            for value in network.parameters()
            if value.requires_grad
        ),
        "gflops": float(get_flops(network, imgsz=config.imgsz)),
        "model_size_bytes": best.stat().st_size,
        "detect_strides": [float(value) for value in network.stride],
    }
    write_json(config.run_dir / "complexity.json", complexity)
    best_epoch = _best_epoch(config)
    write_json(config.run_dir / "best_epoch_summary.json", best_epoch)
    present, missing = _artifact_inventory(config.run_dir)
    environment = environment_record()
    manifest = {
        "schema_version": 1,
        "run_id": config.run_id,
        "seed": config.seed,
        "paper_aliases": config.spec["paper_aliases"],
        "status": "completed",
        "model_yaml": config.spec["model_yaml"],
        "model_yaml_sha256": sha256_file(config.model_yaml),
        "initialization_weight": config.initialization_weight,
        "staged_checkpoint_used": False,
        "official_transfer": json.loads(
            (
                config.run_dir
                / "protocol"
                / "pretrained_transfer_report.json"
            ).read_text(encoding="utf-8")
        ),
        "data_yaml_sha256": sha256_file(config.local_yaml),
        "git_commit": environment["git_commit"],
        "environment": environment,
        "training": config.training,
        "best_epoch": best_epoch,
        "validation_metrics": validation,
        "per_image_evaluation": {
            "split": "val",
            "images": per_image["images"],
            "confidence_threshold": per_image["confidence_threshold"],
            "matching_iou_threshold": per_image["matching_iou_threshold"],
        },
        "complexity": complexity,
        "artifacts_present": present,
        "artifacts_missing": missing,
        "test_used_for_selection": False,
        "completed_at": utc_now(),
    }
    _seal_run(config, manifest, mirror=mirror)
    return manifest


def print_run_banner(config: FormalRunConfig) -> None:
    """在训练开始前打印本次实际生效的固定配置。"""

    effective_training = {
        key: value
        for key, value in config.training.items()
        if key not in {"seeds", "stability_targets", "remaining_arguments"}
    }
    effective_training["seed"] = config.seed
    print("=" * 72)
    print("实验编号：", config.run_id)
    print("随机种子：", config.seed)
    print("模型 YAML：", config.model_yaml)
    print("本地数据 YAML：", config.local_yaml)
    print("官方初始化权重：", config.initialization_weight)
    print("固定 Git 提交：", git_output("rev-parse", "HEAD"))
    print("本次实际训练参数：")
    print(yaml.safe_dump(effective_training, sort_keys=False))
    print("Colab 本地输出：", config.run_dir)
    print("Google Drive 输出：", config.drive_dir)
    print("=" * 72)


__all__ = [
    "EXPECTED_RUN_ARTIFACTS",
    "FormalRunConfig",
    "audit_model",
    "build_and_initialize",
    "capture_environment",
    "enforce_environment_lock",
    "finalize_run",
    "prepare_dataset",
    "prepare_experiment",
    "print_run_banner",
    "resolve_run_state",
    "snapshot_repository",
    "train_foreground",
    "transfer_official_weights",
]
