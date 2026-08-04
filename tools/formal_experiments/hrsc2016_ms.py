"""Prepare the frozen HRSC2016-MS YOLO archive for Colab training.

The source archive remains read-only in Google Drive.  This module copies the
single archive to Colab local storage with byte progress, extracts it safely,
reconstructs a runtime data YAML, and executes the repository's external
dataset audit before a formal S00/S01 run is allowed to start.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml
from tqdm.auto import tqdm

from tools.validate_external_dataset import validate as validate_external_dataset


IMAGE_SUFFIXES = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
EXPECTED_SPLIT_IMAGES = {"train": 610, "val": 460, "test": 610}
EXPECTED_SPLIT_INSTANCES = {"train": 2453, "val": 1953, "test": 3249}
HRSC2016_MS_CITATION = (
    "Chen, W.; Han, B.; Yang, Z.; Gao, X. MSSDet: Multi-Scale "
    "Ship-Detection Framework in Optical Remote-Sensing Images and New "
    "Benchmark. Remote Sensing 2022, 14(21), 5460. "
    "https://doi.org/10.3390/rs14215460"
)
HRSC2016_MS_LICENSE_NOTE = (
    "Research use under the HRSC2016-MS source archive terms. Dataset bytes "
    "remain outside Git and are not redistributed by this repository."
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path, *, show_progress: bool, description: str) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    with (
        path.open("rb") as stream,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=description,
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress,
    ):
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
            progress.update(len(block))
    return digest.hexdigest()


def copy_archive_to_local(
    source: str | Path,
    destination: str | Path,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Copy one Drive archive locally with byte progress and atomic replace."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"HRSC2016-MS archive does not exist: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source_path.stat()
    state_path = destination_path.with_suffix(destination_path.suffix + ".copy.json")
    previous: dict[str, Any] = {}
    if state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
    reusable = (
        destination_path.is_file()
        and destination_path.stat().st_size == source_stat.st_size
        and previous.get("source_size") == source_stat.st_size
        and previous.get("source_mtime_ns") == source_stat.st_mtime_ns
        and isinstance(previous.get("archive_sha256"), str)
    )
    if reusable:
        print(f"复用已经完整复制的本地 ZIP：{destination_path}", flush=True)
        return {
            **previous,
            "source": str(source_path),
            "destination": str(destination_path),
            "copied": False,
        }

    temporary = destination_path.with_suffix(destination_path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    digest = hashlib.sha256()
    copied = 0
    with (
        source_path.open("rb") as source_stream,
        temporary.open("wb") as destination_stream,
        tqdm(
            total=source_stat.st_size,
            unit="B",
            unit_scale=True,
            desc="复制 HRSC2016-MS ZIP 到 Colab",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress,
    ):
        while True:
            block = source_stream.read(16 * 1024 * 1024)
            if not block:
                break
            destination_stream.write(block)
            digest.update(block)
            copied += len(block)
            progress.update(len(block))
        destination_stream.flush()
        os.fsync(destination_stream.fileno())
    if copied != source_stat.st_size:
        temporary.unlink(missing_ok=True)
        raise IOError(
            f"ZIP copy size mismatch: copied={copied}, expected={source_stat.st_size}"
        )
    os.replace(temporary, destination_path)
    report = {
        "source": str(source_path),
        "destination": str(destination_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "archive_sha256": digest.hexdigest(),
        "copied": True,
    }
    _write_json(state_path, report)
    return report


def _safe_member_path(root: Path, filename: str) -> Path:
    normalized = filename.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe ZIP member path: {filename}")
    destination = root.joinpath(*member.parts)
    root_resolved = root.resolve()
    destination_resolved = destination.resolve()
    if destination_resolved != root_resolved and root_resolved not in destination_resolved.parents:
        raise ValueError(f"ZIP member escapes extraction root: {filename}")
    return destination


def _remove_incomplete_extract_root(root: Path) -> None:
    resolved = root.resolve()
    protected = {Path("/").resolve(), Path("/content").resolve(), Path.home().resolve()}
    if resolved in protected or len(resolved.parts) < 3:
        raise ValueError(f"Refusing to remove broad extraction path: {resolved}")
    shutil.rmtree(resolved)


def extract_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    archive_sha256: str,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Safely extract the archive, reusing only a matching completed extraction."""

    archive_path = Path(archive)
    destination_path = Path(destination)
    marker = destination_path.parent / f".{destination_path.name}.extract_complete.json"
    existing: dict[str, Any] = {}
    if marker.is_file():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    reusable = (
        destination_path.is_dir()
        and existing.get("archive_sha256") == archive_sha256
        and existing.get("archive_size") == archive_path.stat().st_size
    )
    if reusable:
        print(f"复用已经完整解压的数据集：{destination_path}", flush=True)
        return {**existing, "extracted": False}

    if destination_path.exists():
        _remove_incomplete_extract_root(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as bundle:
            members = [item for item in bundle.infolist() if not item.is_dir()]
            if not members:
                raise ValueError(f"ZIP contains no files: {archive_path}")
            total_bytes = sum(item.file_size for item in members)
            with (
                tqdm(
                    total=len(members),
                    unit="file",
                    desc="解压文件",
                    dynamic_ncols=True,
                    disable=not show_progress,
                ) as file_progress,
                tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc="解压字节",
                    dynamic_ncols=True,
                    disable=not show_progress,
                ) as byte_progress,
            ):
                for member in members:
                    output = _safe_member_path(destination_path, member.filename)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_suffix(output.suffix + ".part")
                    with bundle.open(member, "r") as source, temporary.open("wb") as target:
                        while True:
                            block = source.read(8 * 1024 * 1024)
                            if not block:
                                break
                            target.write(block)
                            byte_progress.update(len(block))
                    os.replace(temporary, output)
                    file_progress.update(1)
    except Exception:
        if destination_path.exists():
            _remove_incomplete_extract_root(destination_path)
        marker.unlink(missing_ok=True)
        raise
    report = {
        "archive": str(archive_path),
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "destination": str(destination_path),
        "members": len(members),
        "uncompressed_bytes": total_bytes,
        "extracted": True,
    }
    _write_json(marker, report)
    return report


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    index = len(lowered) - 1 - lowered[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _write_audit_csv(path: Path, report: Mapping[str, Any]) -> None:
    fields = (
        "split",
        "images",
        "labels",
        "instances",
        "empty_labels",
        "missing_labels",
        "invalid_rows",
        "corrupt_images",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, values in report["splits"].items():
            writer.writerow(
                {
                    "split": split,
                    **{
                        key: len(values[key]) if isinstance(values[key], list) else values[key]
                        for key in fields[1:]
                    },
                }
            )


def _assert_expected_dataset(
    root: Path,
    *,
    expected_images: Mapping[str, int],
    expected_instances: Mapping[str, int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"HRSC2016-MS split directories are incomplete: {image_dir}, {label_dir}"
            )
        images = sorted(
            item for item in image_dir.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(label_dir.rglob("*.txt"))
        image_stems = {item.stem for item in images}
        label_stems = {item.stem for item in labels}
        if image_stems != label_stems:
            raise ValueError(
                f"{split} image/label mismatch: missing_labels="
                f"{sorted(image_stems - label_stems)[:10]}, orphan_labels="
                f"{sorted(label_stems - image_stems)[:10]}"
            )
        instances = 0
        empty_labels = 0
        for image in images:
            label = _label_for_image(image)
            lines = [
                line.strip()
                for line in label.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except (ValueError, IndexError) as error:
                    raise ValueError(f"Invalid label row {label}:{line_number}: {line}") from error
                valid = (
                    len(fields) == 5
                    and class_id == 0
                    and all(math.isfinite(value) for value in values)
                    and all(0.0 <= value <= 1.0 for value in values)
                    and values[2] > 0.0
                    and values[3] > 0.0
                )
                if not valid:
                    raise ValueError(f"Invalid label row {label}:{line_number}: {line}")
                instances += 1
        if len(images) != int(expected_images[split]):
            raise ValueError(
                f"{split} image count changed: {len(images)} != {expected_images[split]}"
            )
        if instances != int(expected_instances[split]):
            raise ValueError(
                f"{split} instance count changed: {instances} != {expected_instances[split]}"
            )
        split_ids[split] = image_stems
        summary["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": instances,
            "empty_labels": empty_labels,
        }
    overlaps = {
        "train_val": sorted(split_ids["train"] & split_ids["val"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "val_test": sorted(split_ids["val"] & split_ids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Split ID overlap found: {overlaps}")
    summary["split_id_overlap"] = overlaps
    return summary


def prepare_hrsc2016_ms_archive(
    archive_path: str | Path,
    *,
    local_archive_path: str | Path = "/content/dataset_archives/HRSC2016_MS_YOLO.zip",
    extract_root: str | Path = "/content/ship_detection/HRSC2016_MS_YOLO",
    runtime_yaml: str | Path = "/content/ship_detection/hrsc2016_ms_runtime.yaml",
    descriptor_path: str | Path = "/content/ship_detection/hrsc2016_ms_descriptor.yaml",
    audit_output: str | Path = "/content/ship_detection/hrsc2016_ms_integration_audit.json",
    artifact_dir: str | Path | None = None,
    expected_images: Mapping[str, int] = EXPECTED_SPLIT_IMAGES,
    expected_instances: Mapping[str, int] = EXPECTED_SPLIT_INSTANCES,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Prepare and audit the frozen YOLO-HBB HRSC2016-MS archive."""

    archive_path = Path(archive_path)
    local_archive_path = Path(local_archive_path)
    extract_root = Path(extract_root)
    runtime_yaml = Path(runtime_yaml)
    descriptor_path = Path(descriptor_path)
    audit_output = Path(audit_output)
    copy_report = copy_archive_to_local(
        archive_path,
        local_archive_path,
        show_progress=show_progress,
    )
    archive_sha256 = str(copy_report["archive_sha256"])
    if not archive_sha256:
        archive_sha256 = _sha256(
            local_archive_path,
            show_progress=show_progress,
            description="校验本地 ZIP",
        )
    extraction_report = extract_archive(
        local_archive_path,
        extract_root,
        archive_sha256=archive_sha256,
        show_progress=show_progress,
    )
    expected_summary = _assert_expected_dataset(
        extract_root,
        expected_images=expected_images,
        expected_instances=expected_instances,
    )

    runtime_payload = {
        "path": str(extract_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "ship"},
    }
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(
        yaml.safe_dump(runtime_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    descriptor_payload = {
        "dataset_name": "HRSC2016-MS YOLO-HBB",
        "root": str(extract_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "class_mapping": {0: "ship"},
        "annotation_format": "yolo_detection",
        "license": HRSC2016_MS_LICENSE_NOTE,
        "citation": HRSC2016_MS_CITATION,
        "image_resolution": "variable; source images retain their original resolution",
        "notes": (
            "Frozen official HRSC2016-MS split. Horizontal boxes were converted "
            "from the XML bndbox annotations; rotated boxes are not used by this "
            "YOLO detect experiment."
        ),
        "source_archive": str(archive_path),
        "source_archive_sha256": archive_sha256,
        "expected_split_images": dict(expected_images),
        "expected_split_instances": dict(expected_instances),
    }
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text(
        yaml.safe_dump(descriptor_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print("执行图像可读性、标签合法性和跨划分重复审计……", flush=True)
    audit = validate_external_dataset(descriptor_path)
    if not audit["passed"]:
        raise ValueError(f"HRSC2016-MS integration audit failed: {audit['errors']}")
    for split in ("train", "val", "test"):
        values = audit["splits"][split]
        if values["images"] != int(expected_images[split]):
            raise ValueError(f"Unexpected audited image count for {split}: {values['images']}")
        if values["instances"] != int(expected_instances[split]):
            raise ValueError(
                f"Unexpected audited instance count for {split}: {values['instances']}"
            )
    _write_json(audit_output, audit)
    _write_audit_csv(audit_output.with_suffix(".csv"), audit)
    manifest = {
        "dataset_id": "hrsc2016_ms_yolo_hbb_v1",
        "source_read_only": True,
        "source_archive": str(archive_path),
        "local_archive": str(local_archive_path),
        "archive_size": local_archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "copy": copy_report,
        "extraction": extraction_report,
        "data_yaml": str(runtime_yaml),
        "descriptor": str(descriptor_path),
        "audit": str(audit_output),
        "summary": expected_summary,
        "citation": HRSC2016_MS_CITATION,
        "test_used_for_model_selection": False,
    }
    manifest_path = audit_output.with_name("hrsc2016_ms_integration_manifest.json")
    _write_json(manifest_path, manifest)

    if artifact_dir is not None:
        destination = Path(artifact_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for source in (
            runtime_yaml,
            descriptor_path,
            audit_output,
            audit_output.with_suffix(".csv"),
            manifest_path,
        ):
            shutil.copyfile(source, destination / source.name)

    print("HRSC2016-MS 数据集审计通过：", expected_summary["splits"], flush=True)
    return {
        "root": extract_root,
        "data_yaml": runtime_yaml,
        "descriptor": descriptor_path,
        "audit": audit_output,
        "manifest": manifest_path,
        "summary": expected_summary,
        "archive_sha256": archive_sha256,
    }


__all__ = [
    "EXPECTED_SPLIT_IMAGES",
    "EXPECTED_SPLIT_INSTANCES",
    "HRSC2016_MS_CITATION",
    "copy_archive_to_local",
    "extract_archive",
    "prepare_hrsc2016_ms_archive",
]
