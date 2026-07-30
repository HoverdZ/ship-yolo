"""Load and validate the canonical formal-experiment registry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "experiments" / "formal_experiment_registry.yaml"

REQUIRED_FIELDS = {
    "experiment_id",
    "canonical_run_id",
    "paper_aliases",
    "base_model",
    "model_yaml",
    "initialization_weight",
    "dataset_id",
    "data_yaml",
    "imgsz",
    "epochs",
    "batch",
    "optimizer",
    "seed",
    "module_flags",
    "expected_detect_strides",
    "output_drive_dir",
    "notebook_path",
    "status",
    "artifact_manifest",
    "notes",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping: {source}")
    return payload


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry = load_yaml(path)
    validate_registry(registry)
    return registry


def model_yaml_sha256(run: dict[str, Any]) -> str:
    path = ROOT / run["model_yaml"]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    runs = registry.get("canonical_runs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError("Registry must define a non-empty canonical_runs mapping.")
    aliases: dict[str, str] = {}
    output_dirs: dict[str, str] = {}
    errors: list[str] = []
    for run_id, run in runs.items():
        if not isinstance(run, dict):
            errors.append(f"{run_id}: run entry is not a mapping")
            continue
        missing = sorted(REQUIRED_FIELDS - set(run))
        if missing:
            errors.append(f"{run_id}: missing fields {missing}")
        if run.get("experiment_id") != run_id:
            errors.append(
                f"{run_id}: experiment_id={run.get('experiment_id')!r}"
            )
        if run.get("canonical_run_id") != run_id:
            errors.append(
                f"{run_id}: canonical_run_id={run.get('canonical_run_id')!r}"
            )
        model_path = ROOT / str(run.get("model_yaml", ""))
        if not model_path.is_file():
            errors.append(f"{run_id}: model YAML missing: {model_path}")
        notebook = Path(str(run.get("notebook_path", "")))
        if notebook.suffix != ".ipynb":
            errors.append(f"{run_id}: invalid Notebook path: {notebook}")
        strides = run.get("expected_detect_strides")
        if strides not in ([8, 16, 32], [4, 8, 16]):
            errors.append(f"{run_id}: unsupported detect strides: {strides}")
        for alias in run.get("paper_aliases", []):
            if alias in aliases:
                errors.append(
                    f"Alias {alias} appears in both {aliases[alias]} and {run_id}"
                )
            aliases[alias] = run_id
        output = str(run.get("output_drive_dir", ""))
        if output in output_dirs:
            errors.append(
                f"Output {output} shared by {output_dirs[output]} and {run_id}"
            )
        output_dirs[output] = run_id
        if Path(output).is_absolute() or ".." in Path(output).parts:
            errors.append(f"{run_id}: output_drive_dir must be safe and relative")
        manifest = str(run.get("artifact_manifest", ""))
        if not manifest.startswith(output.rstrip("/") + "/"):
            errors.append(
                f"{run_id}: manifest is outside its output directory"
            )
    if errors:
        raise ValueError("Invalid experiment registry:\n- " + "\n- ".join(errors))
    return {
        "canonical_runs": len(runs),
        "paper_aliases": len(aliases),
        "unique_output_dirs": len(output_dirs),
    }


def resolve_run(
    run_or_alias: str,
    registry: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    registry = registry or load_registry()
    runs = registry["canonical_runs"]
    if run_or_alias in runs:
        return run_or_alias, runs[run_or_alias]
    matches = [
        (run_id, run)
        for run_id, run in runs.items()
        if run_or_alias in run["paper_aliases"]
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown or ambiguous run/alias: {run_or_alias}")
    return matches[0]


__all__ = [
    "DEFAULT_REGISTRY",
    "ROOT",
    "load_registry",
    "load_yaml",
    "model_yaml_sha256",
    "resolve_run",
    "validate_registry",
]
