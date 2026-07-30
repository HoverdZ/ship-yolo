"""Checks for the bilingual experiment skeleton and safe result updater."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tools.build_experiment_manuscript_skeleton import (
    RUNS,
    SECTIONS,
    TABLES,
    build_markdown,
    placeholder_payload,
)
from tools.update_manuscript_experiment_results import (
    collect_replacements,
    update_documents,
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\{\{PENDING_[A-Z0-9_]+\}\}", text))


def test_bilingual_skeleton_has_matching_numbered_evidence_slots() -> None:
    assert len(SECTIONS) == 13
    assert list(TABLES) == list(range(1, 13))
    figure_numbers = [
        number
        for section in SECTIONS
        for number, _zh, _en in section["figures"]
    ]
    assert figure_numbers == list(range(1, 10))
    chinese = build_markdown("zh")
    english = build_markdown("en")
    assert _tokens(chinese) == _tokens(english)
    assert len(_tokens(chinese)) >= 100
    for index in range(1, 14):
        assert f"4.{index} " in chinese
        assert f"4.{index} " in english
    for index in range(1, 13):
        assert f"表 {index}" in chinese
        assert f"Table {index}" in english
    for index in range(1, 10):
        assert f"图 {index}" in chinese
        assert f"Figure {index}" in english


def test_placeholder_registry_covers_every_canonical_run() -> None:
    payload = placeholder_payload()
    assert payload["runs"] == RUNS
    assert payload["policy"]["replace_only_from_real_artifacts"] is True
    assert payload["policy"]["directional_conclusions_pre_authored"] is False


def test_updater_is_dry_run_by_default_and_backs_up_on_apply(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    manifest = run_root / "R00" / "seed_0" / "run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "run_id": "R00",
                "seed": 0,
                "status": "completed",
                "validation_metrics": {
                    "precision": 0.812346,
                    "recall": 0.712345,
                    "map50": 0.772345,
                    "map75": 0.312345,
                    "map50_95": 0.322346,
                },
                "complexity": {
                    "parameters": 2624080,
                    "gflops": 6.5,
                },
            }
        ),
        encoding="utf-8",
    )
    document = tmp_path / "draft.md"
    latex = tmp_path / "draft.tex"
    original = (
        "P={{PENDING_R00_PRECISION}}; "
        "AP={{PENDING_R00_MAP5095}}; "
        "FPS={{PENDING_R00_FPS}}\n"
    )
    document.write_text(original, encoding="utf-8")
    latex_original = (
        r"P=\{\{PENDING\_R00\_PRECISION\}\}; "
        r"AP=\{\{PENDING\_R00\_MAP5095\}\}; "
        r"FPS=\{\{PENDING\_R00\_FPS\}\}" + "\n"
    )
    latex.write_text(latex_original, encoding="utf-8")
    replacements, provenance = collect_replacements(run_root)
    assert provenance[0]["run_id"] == "R00"

    dry_backup = tmp_path / "dry-backup"
    records, _diffs = update_documents(
        [document, latex],
        replacements,
        apply=False,
        backup_root=dry_backup,
    )
    assert records[0]["replacement_count"] == 2
    assert records[1]["replacement_count"] == 2
    assert document.read_text(encoding="utf-8") == original
    assert latex.read_text(encoding="utf-8") == latex_original
    assert not dry_backup.exists()

    backup = tmp_path / "backup"
    update_documents(
        [document, latex],
        replacements,
        apply=True,
        backup_root=backup,
    )
    updated = document.read_text(encoding="utf-8")
    assert "P=0.81235" in updated
    assert "AP=0.32235" in updated
    assert "{{PENDING_R00_FPS}}" in updated
    assert (backup / document.name).read_text(encoding="utf-8") == original
    latex_updated = latex.read_text(encoding="utf-8")
    assert "P=0.81235" in latex_updated
    assert "AP=0.32235" in latex_updated
    assert r"\{\{PENDING\_R00\_FPS\}\}" in latex_updated
    assert (backup / latex.name).read_text(encoding="utf-8") == latex_original
