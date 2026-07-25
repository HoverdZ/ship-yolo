"""Google Colab helpers for the YOLO11n C3Cross-P23 screening experiment."""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm

from tools.c3cross_p23_workflow import (
    EXPERIMENT_NAME,
    ap75_audit,
    hybrid_initialize,
    run_p23_screening,
    run_winner_finetune,
    structure_check,
    summarize_finetune_run,
    summarize_screening_run,
    validate_checkpoint_metrics,
)

REPO_REF = "experiment/yolo11n-c3cross-p23"
REPO_ROOT = Path("/content/ship-yolo")
DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/ship_detection/data")
LOCAL_DATA_ROOT = Path("/content/ship_detection/data")
LOCAL_DATA_YAML = Path("/content/ship_detection/data_local.yaml")
DRIVE_RUNS_ROOT = Path("/content/drive/MyDrive/ship_detection/runs")
DRIVE_AUDIT_ROOT = Path("/content/drive/MyDrive/ship_detection/audits")

# Colab commonly exposes only two CPU cores. Deriving this from os.cpu_count()
# previously reduced the pool to four workers and made Drive I/O unnecessarily
# slow. Sixteen concurrent copyfile calls is the established notebook default.
COPY_WORKERS = 16


def _copy_one(pair: tuple[Path, Path]) -> tuple[int, bool]:
    source, destination = pair
    size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        return size, False
    shutil.copyfile(source, destination)
    return size, True


def copy_dataset_to_local(
    source_root: Path = DRIVE_DATA_ROOT,
    destination_root: Path = LOCAL_DATA_ROOT,
    workers: int = COPY_WORKERS,
) -> dict[str, Any]:
    """Copy Drive data with parallel copyfile calls and file/byte progress."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Google Drive dataset not found: {source_root}")
    source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"No dataset files found under: {source_root}")

    total_bytes = sum(path.stat().st_size for path in source_files)
    jobs = [
        (source, destination_root / source.relative_to(source_root))
        for source in source_files
    ]
    copied_files = 0
    copied_bytes = 0
    with (
        tqdm(total=len(jobs), unit="file", desc="Files") as file_bar,
        tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Bytes") as byte_bar,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        for size, copied in executor.map(_copy_one, jobs, chunksize=8):
            file_bar.update(1)
            byte_bar.update(size)
            if copied:
                copied_files += 1
                copied_bytes += size
            file_bar.set_postfix(
                copied=copied_files,
                copied_bytes=f"{copied_bytes:,}",
                workers=workers,
                refresh=False,
            )

    local_files = [path for path in destination_root.rglob("*") if path.is_file()]
    yaml_files = sorted(
        path
        for path in local_files
        if path.suffix.lower() in {".yaml", ".yml"}
    )
    inventory = {
        "source_files": len(source_files),
        "local_files": len(local_files),
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "workers": workers,
        "yaml_files": [str(path) for path in yaml_files],
    }
    print("Local dataset inventory:", inventory)
    if not yaml_files:
        raise FileNotFoundError("No dataset YAML exists in the local copy.")
    return inventory


def create_local_data_yaml(relative_path: str = "data.yaml") -> Path:
    """Create a local-path YAML without enforcing fixed file-count equality."""
    source_yaml = LOCAL_DATA_ROOT / relative_path
    if not source_yaml.is_file():
        candidates = sorted(
            path
            for path in LOCAL_DATA_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
        raise FileNotFoundError(
            f"Configured dataset YAML does not exist: {source_yaml}. "
            f"Candidates: {[str(path) for path in candidates]}"
        )
    original = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    required = {"train", "val", "test", "nc", "names"}
    missing = sorted(required.difference(original))
    if missing:
        raise KeyError(f"Dataset YAML is missing required keys: {missing}")
    local = dict(original)
    local["path"] = str(LOCAL_DATA_ROOT)
    LOCAL_DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_YAML.write_text(
        yaml.safe_dump(local, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Local dataset YAML: {LOCAL_DATA_YAML}")
    print("Validation and test entries were preserved without augmentation.")
    return LOCAL_DATA_YAML


def print_environment(commit: str) -> None:
    import ultralytics

    if ultralytics.__version__ != "8.4.92":
        raise RuntimeError(
            f"Ultralytics must be 8.4.92, got {ultralytics.__version__}"
        )
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    print("Python:", sys.version)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", gpu)
    print("Ultralytics:", ultralytics.__version__)
    print("Repository commit:", commit)
    print("Experiment:", EXPERIMENT_NAME)
    print("Dataset copy workers:", COPY_WORKERS)


__all__ = [
    "COPY_WORKERS",
    "DRIVE_AUDIT_ROOT",
    "DRIVE_RUNS_ROOT",
    "EXPERIMENT_NAME",
    "LOCAL_DATA_YAML",
    "REPO_REF",
    "ap75_audit",
    "copy_dataset_to_local",
    "create_local_data_yaml",
    "hybrid_initialize",
    "print_environment",
    "run_p23_screening",
    "run_winner_finetune",
    "structure_check",
    "summarize_finetune_run",
    "summarize_screening_run",
    "validate_checkpoint_metrics",
]
