"""Tests for the HRSC2016-MS archive-to-training integration."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from tools.formal_experiments.hrsc2016_ms import (
    extract_archive,
    prepare_hrsc2016_ms_archive,
)


def _build_tiny_dataset(root: Path) -> None:
    colors = {"train": (255, 0, 0), "val": (0, 255, 0), "test": (0, 0, 255)}
    labels = {
        "train": ["0 0.5 0.5 0.25 0.25"],
        "val": ["0 0.4 0.4 0.20 0.20", "0 0.7 0.7 0.10 0.10"],
        "test": ["0 0.6 0.6 0.30 0.30"],
    }
    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("RGB", (12, 10), colors[split]).save(image_dir / f"{split}.bmp")
        (label_dir / f"{split}.txt").write_text(
            "\n".join(labels[split]) + "\n",
            encoding="utf-8",
        )
    (root / "data.yaml").write_text(
        "path: /obsolete/runtime/path\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n  0: ship\n",
        encoding="utf-8",
    )


def _zip_tree(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(source).as_posix())


def test_prepare_archive_preserves_frozen_split_and_reuses_completed_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _build_tiny_dataset(source)
    archive = tmp_path / "HRSC2016_MS_YOLO.zip"
    _zip_tree(source, archive)
    parameters = {
        "archive_path": archive,
        "local_archive_path": tmp_path / "local" / archive.name,
        "extract_root": tmp_path / "extracted",
        "runtime_yaml": tmp_path / "runtime.yaml",
        "descriptor_path": tmp_path / "descriptor.yaml",
        "audit_output": tmp_path / "audit.json",
        "artifact_dir": tmp_path / "artifacts",
        "expected_images": {"train": 1, "val": 1, "test": 1},
        "expected_instances": {"train": 1, "val": 2, "test": 1},
        "show_progress": False,
    }
    first = prepare_hrsc2016_ms_archive(**parameters)
    second = prepare_hrsc2016_ms_archive(**parameters)
    assert first["summary"] == second["summary"]
    assert first["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert first["data_yaml"].read_text(encoding="utf-8").startswith(
        f"path: {Path(parameters['extract_root']).resolve()}"
    )
    assert (tmp_path / "artifacts" / "hrsc2016_ms_integration_manifest.json").is_file()
    assert (tmp_path / "artifacts" / "audit.csv").is_file()


def test_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "unsafe")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Unsafe ZIP member path"):
        extract_archive(
            archive,
            tmp_path / "extract" / "dataset",
            archive_sha256=digest,
            show_progress=False,
        )
    assert not (tmp_path / "extract" / "escape.txt").exists()
