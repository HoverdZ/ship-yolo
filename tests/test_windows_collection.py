"""Windows collection helpers must preserve Unicode paths and never overwrite."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.windows_collection import (
    copy_tree,
    destination_version,
    sha256,
    verify_checksum_manifest,
)


def test_unicode_copy_resume_and_checksum(tmp_path: Path) -> None:
    source = tmp_path / "云端实验"
    source.mkdir()
    (source / "结果.csv").write_text("metric,value\nmAP,1\n", encoding="utf-8")
    destination = destination_version(
        tmp_path / "遥感船舶检测论文",
        "20260730_120000",
        allow_resume=False,
    )
    first = copy_tree(source, destination)
    second = copy_tree(source, destination)
    assert first[0]["status"] == "copied"
    assert second[0]["status"] == "verified_skip"
    copied = destination / "结果.csv"
    manifest = destination / "artifact_checksums.sha256"
    manifest.write_text(
        f"{sha256(copied)}  结果.csv\n",
        encoding="utf-8",
    )
    assert all(row["passed"] for row in verify_checksum_manifest(manifest))


def test_different_existing_file_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "same.txt").write_text("source", encoding="utf-8")
    (destination / "same.txt").write_text("user", encoding="utf-8")
    with pytest.raises(FileExistsError):
        copy_tree(source, destination)
    assert (destination / "same.txt").read_text(encoding="utf-8") == "user"
