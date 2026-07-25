"""Google Colab support for cumulative DySample/PLS/SCAM experiments.

The notebook imports these helpers from the checked-out repository so module
definitions and training logic are never copied into disposable Colab cells.
"""

from __future__ import annotations

import concurrent.futures
import shutil
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm

DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/ship_detection/data")
LOCAL_DATA_ROOT = Path("/content/ship_detection/data")
LOCAL_DATA_YAML = Path("/content/ship_detection/data_local.yaml")
DRIVE_RUNS_ROOT = Path("/content/drive/MyDrive/ship_detection/runs")
DRIVE_AUDIT_ROOT = Path("/content/drive/MyDrive/ship_detection/audits")
COPY_WORKERS = 16


def _copy_one(job: tuple[Path, Path]) -> tuple[int, bool]:
    source, destination = job
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
    """Copy Drive data concurrently with separate file and byte progress."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Drive dataset not found: {source_root}")
    source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"No files found under Drive dataset: {source_root}")

    jobs = [
        (source, destination_root / source.relative_to(source_root))
        for source in source_files
    ]
    total_bytes = sum(source.stat().st_size for source in source_files)
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
                workers=workers,
                refresh=False,
            )

    inventory = {
        "source_files": len(source_files),
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "workers": workers,
        "destination": str(destination_root),
    }
    print("Dataset copy complete:", inventory)
    return inventory


def create_local_data_yaml(relative_path: str = "data.yaml") -> Path:
    """Point the copied YAML at local storage without comparing fixed counts."""
    source_yaml = LOCAL_DATA_ROOT / relative_path
    if not source_yaml.is_file():
        candidates = sorted(
            path
            for path in LOCAL_DATA_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
        raise FileNotFoundError(
            f"Dataset YAML not found: {source_yaml}. "
            f"Candidates: {[str(path) for path in candidates]}"
        )
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    missing = sorted({"train", "val", "test", "nc", "names"} - set(payload))
    if missing:
        raise KeyError(f"Dataset YAML is missing required keys: {missing}")

    local = dict(payload)
    local["path"] = str(LOCAL_DATA_ROOT)
    LOCAL_DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_YAML.write_text(
        yaml.safe_dump(local, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Local dataset YAML: {LOCAL_DATA_YAML}")
    print("train/val/test entries are unchanged; no validation/test augmentation is added.")
    return LOCAL_DATA_YAML


__all__ = [
    "COPY_WORKERS",
    "DRIVE_AUDIT_ROOT",
    "DRIVE_RUNS_ROOT",
    "LOCAL_DATA_YAML",
    "copy_dataset_to_local",
    "create_local_data_yaml",
]
