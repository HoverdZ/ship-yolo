"""Safe, resumable copy helpers for Windows paper-artifact collection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version_name(value: str | None = None) -> str:
    return value or datetime.now().strftime("%Y%m%d_%H%M%S")


def destination_version(
    root: str | Path,
    version: str | None,
    *,
    allow_resume: bool,
) -> Path:
    destination = Path(root).expanduser().resolve() / version_name(version)
    if destination.exists() and not allow_resume:
        raise FileExistsError(
            f"Version directory already exists; choose another --version: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def copy_one(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if destination.is_file():
        if destination.stat().st_size == source.stat().st_size and sha256(destination) == source_hash:
            return {
                "source": str(source),
                "destination": str(destination),
                "bytes": source.stat().st_size,
                "sha256": source_hash,
                "status": "verified_skip",
            }
        raise FileExistsError(
            f"Refusing to overwrite a different existing file: {destination}"
        )
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copyfile(source, temporary)
    copied_hash = sha256(temporary)
    if copied_hash != source_hash:
        raise IOError(f"SHA256 mismatch while copying {source}")
    os.replace(temporary, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": source.stat().st_size,
        "sha256": source_hash,
        "status": "copied",
    }


def copy_tree(
    source_root: str | Path,
    destination_root: str | Path,
) -> list[dict[str, Any]]:
    source = Path(source_root).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == destination or source in destination.parents:
        raise ValueError("Destination must not be inside the source tree.")
    rows = []
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        rows.append(copy_one(path, destination / path.relative_to(source)))
    return rows


def write_report(
    destination: Path,
    rows: Iterable[dict[str, Any]],
    *,
    source: str,
) -> Path:
    values = list(rows)
    report = {
        "source": source,
        "destination": str(destination),
        "files": len(values),
        "bytes": sum(int(row["bytes"]) for row in values),
        "copied": sum(row["status"] == "copied" for row in values),
        "verified_skips": sum(
            row["status"] == "verified_skip" for row in values
        ),
        "records": values,
    }
    path = destination / "collection_report.json"
    if path.exists():
        # A resumed collection must preserve the prior report.
        path = destination / (
            "collection_report_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".json"
        )
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def verify_checksum_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    root = manifest.parent
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = root / relative
        actual = sha256(target) if target.is_file() else None
        rows.append(
            {
                "manifest": str(manifest),
                "path": relative,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        )
    return rows


__all__ = [
    "copy_one",
    "copy_tree",
    "destination_version",
    "sha256",
    "verify_checksum_manifest",
    "version_name",
    "write_report",
]
