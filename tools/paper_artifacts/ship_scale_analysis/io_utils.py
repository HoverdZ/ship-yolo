"""Stable serialization and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path, *, paper_rounding: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.2f" if paper_rounding else "%.12g",
    )


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    excluded = {"artifact_manifest.json", "artifact_checksums.sha256"}
    artifacts = []
    for path in sorted((item for item in output_dir.rglob("*") if item.is_file()), key=lambda item: item.relative_to(output_dir).as_posix()):
        relative = path.relative_to(output_dir).as_posix()
        if relative in excluded:
            continue
        artifacts.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": 1, "metadata": metadata, "artifacts": artifacts}


def write_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / "artifact_checksums.sha256"
    lines = []
    for path in sorted((item for item in output_dir.rglob("*") if item.is_file()), key=lambda item: item.relative_to(output_dir).as_posix()):
        relative = path.relative_to(output_dir).as_posix()
        if relative == checksum_path.name:
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return checksum_path


def deterministic_zip(source_dir: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((item for item in source_dir.rglob("*") if item.is_file()), key=lambda item: item.relative_to(source_dir).as_posix()):
            relative = f"{source_dir.name}/{path.relative_to(source_dir).as_posix()}"
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return destination


def stable_environment() -> dict[str, str]:
    return {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not required; all ordering is explicitly sorted"),
        "quantile_method": "linear",
    }
