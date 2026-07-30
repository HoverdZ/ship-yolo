"""Shared strict result extraction and multi-format table output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from tools.formal_experiments.registry import load_registry

RESULT_FIELDS = (
    "precision",
    "recall",
    "map50",
    "map75",
    "map50_95",
    "params",
    "gflops",
    "latency_ms",
    "fps",
)


def placeholder(run_id: str, field: str) -> str:
    normalized = field.upper().replace("_", "")
    aliases = {
        "MAP5095": "MAP5095",
        "LATENCYMS": "LATENCY",
    }
    normalized = aliases.get(normalized, normalized)
    return "{{PENDING_" + run_id.upper() + "_" + normalized + "}}"


def _manifest_path(root: Path, run_id: str, seed: int) -> Path:
    candidates = (
        root / run_id / f"seed_{seed}" / "run_manifest.json",
        root / run_id / "run_manifest.json",
        root / f"{run_id}_seed_{seed}" / "run_manifest.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _read_one(root: Path, run_id: str, seed: int) -> dict[str, Any]:
    path = _manifest_path(root, run_id, seed)
    empty = {
        field: placeholder(run_id, field)
        for field in RESULT_FIELDS
    }
    record: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "manifest_path": str(path),
        "status": "pending",
        **empty,
    }
    if not path.is_file():
        return record
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != run_id or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Manifest identity mismatch: {path}")
    validation = payload.get("validation_metrics", {})
    complexity = payload.get("complexity", {})
    latency = complexity.get("latency", {})
    record.update(
        {
            "status": payload.get("status", "unknown"),
            "precision": validation.get(
                "precision", placeholder(run_id, "precision")
            ),
            "recall": validation.get(
                "recall", placeholder(run_id, "recall")
            ),
            "map50": validation.get("map50", placeholder(run_id, "map50")),
            "map75": validation.get("map75", placeholder(run_id, "map75")),
            "map50_95": validation.get(
                "map50_95", placeholder(run_id, "map50_95")
            ),
            "params": complexity.get(
                "parameters", placeholder(run_id, "params")
            ),
            "gflops": complexity.get(
                "gflops", placeholder(run_id, "gflops")
            ),
            "latency_ms": latency.get(
                "mean_ms", placeholder(run_id, "latency_ms")
            ),
            "fps": latency.get("fps", placeholder(run_id, "fps")),
            "git_commit": payload.get("git_commit"),
            "best_epoch": payload.get("best_epoch", {}).get("best_epoch"),
            "initialization_weight": payload.get("initialization_weight"),
            "test_used_for_selection": payload.get(
                "test_used_for_selection"
            ),
        }
    )
    return record


def collect_metrics(
    root: str | Path,
    *,
    run_ids: Iterable[str] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    registry = load_registry()
    selected = list(run_ids or registry["canonical_runs"])
    return [_read_one(Path(root), run_id, seed) for run_id in selected]


def write_rows(
    rows: list[dict[str, Any]],
    output_prefix: str | Path,
    *,
    columns: list[str] | None = None,
    title: str = "Formal experiment results",
) -> dict[str, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = columns or list(rows[0] if rows else ("run_id",))
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path = prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = prefix.with_suffix(".md")
    header = "| " + " | ".join(fields) + " |"
    divider = "|" + "|".join("---" for _ in fields) + "|"
    body = [
        "| "
        + " | ".join(str(row.get(field, "")) for field in fields)
        + " |"
        for row in rows
    ]
    markdown_path.write_text(
        f"# {title}\n\n" + "\n".join((header, divider, *body)) + "\n",
        encoding="utf-8",
    )
    latex_path = prefix.with_suffix(".tex")
    escaped = lambda value: str(value).replace("_", r"\_")
    latex_rows = [
        " & ".join(escaped(row.get(field, "")) for field in fields) + r" \\"
        for row in rows
    ]
    latex_path.write_text(
        "\n".join(
            (
                r"\begin{tabular}{" + "l" * len(fields) + "}",
                r"\toprule",
                " & ".join(escaped(field) for field in fields) + r" \\",
                r"\midrule",
                *latex_rows,
                r"\bottomrule",
                r"\end{tabular}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": markdown_path,
        "latex": latex_path,
    }


def table_rows(
    root: str | Path,
    aliases: list[tuple[str, str]],
    *,
    extra: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    metrics = {
        row["run_id"]: row
        for row in collect_metrics(
            root,
            run_ids=[run_id for _alias, run_id in aliases],
        )
    }
    rows = []
    for alias, run_id in aliases:
        row = {"paper_id": alias, **metrics[run_id]}
        row.update((extra or {}).get(alias, {}))
        rows.append(row)
    return rows


__all__ = [
    "RESULT_FIELDS",
    "collect_metrics",
    "placeholder",
    "table_rows",
    "write_rows",
]
