"""Audit formal checkpoints and pair them with their exact training logs.

The audit deliberately does not infer checkpoint identity from a rounded metric
or a filename alone.  It combines checkpoint metadata, embedded model YAML,
repository YAML matches, module signatures, training arguments, and a numeric
comparison between ``checkpoint['train_results']`` and every candidate CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}

CUSTOM_MODULES = {
    "C3k2_InceptionDW",
    "C3k2_PConv",
    "C3k2_LSKConv",
    "DySample",
    "SCAM",
    "CASCAM",
    "CASCAMFixedBeta",
    "CASCAMUnbounded",
    "VGUPPreprocessor",
    "ERUPPreprocessor",
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalise_yaml_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    if isinstance(value, dict):
        return {
            str(key): _normalise_yaml_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_yaml_value(item) for item in value]
    return value


def architecture_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or "backbone" not in payload or "head" not in payload:
        return None
    return _normalise_yaml_value(
        {
            "scale": payload.get("scale", "n"),
            "scales": payload.get("scales"),
            "backbone": payload["backbone"],
            "head": payload["head"],
        }
    )


def architecture_sha256(payload: dict[str, Any]) -> str | None:
    architecture = architecture_payload(payload)
    if architecture is None:
        return None
    encoded = json.dumps(
        architecture,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def repository_architectures(repo_root: str | Path) -> dict[str, list[str]]:
    root = Path(repo_root)
    matches: dict[str, list[str]] = {}
    for path in sorted((root / "experiments").rglob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        digest = architecture_sha256(payload)
        if digest:
            matches.setdefault(digest, []).append(path.relative_to(root).as_posix())
    return matches


def _read_csv(path: Path) -> dict[str, list[Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    output: dict[str, list[Any]] = {column: [] for column in rows[0]}
    for row in rows:
        for column, value in row.items():
            text = value.strip() if isinstance(value, str) else value
            try:
                output[column].append(float(text))
            except (TypeError, ValueError):
                output[column].append(text)
    return output


def compare_training_results(
    embedded: dict[str, Any] | None,
    candidate: dict[str, list[Any]],
) -> dict[str, Any]:
    if not isinstance(embedded, dict) or not embedded or not candidate:
        return {"comparable": False, "reason": "missing embedded or CSV results"}
    common = sorted(set(embedded).intersection(candidate))
    if not common:
        return {"comparable": False, "reason": "no common columns"}
    lengths = {
        len(embedded[column])
        for column in common
        if hasattr(embedded[column], "__len__")
    }
    lengths.update(len(candidate[column]) for column in common)
    if len(lengths) != 1:
        return {
            "comparable": False,
            "reason": "row count mismatch",
            "lengths": sorted(lengths),
        }
    differences: list[float] = []
    non_numeric_mismatches = 0
    compared_values = 0
    for column in common:
        for left, right in zip(embedded[column], candidate[column], strict=True):
            try:
                left_number = float(left)
                right_number = float(right)
            except (TypeError, ValueError):
                if str(left) != str(right):
                    non_numeric_mismatches += 1
                continue
            if math.isfinite(left_number) and math.isfinite(right_number):
                differences.append(abs(left_number - right_number))
                compared_values += 1
            elif not (math.isnan(left_number) and math.isnan(right_number)):
                non_numeric_mismatches += 1
    maximum = max(differences, default=float("inf"))
    mean = sum(differences) / len(differences) if differences else float("inf")
    exact = maximum <= 1e-10 and non_numeric_mismatches == 0
    rounded_exact = maximum <= 5e-5 and non_numeric_mismatches == 0
    return {
        "comparable": True,
        "columns": common,
        "rows": next(iter(lengths)),
        "compared_numeric_values": compared_values,
        "non_numeric_mismatches": non_numeric_mismatches,
        "max_abs_diff": maximum,
        "mean_abs_diff": mean,
        "exact": exact,
        "rounded_exact": rounded_exact,
    }


def _human_identity(module_counts: Counter[str]) -> str:
    if module_counts["C3k2_PConv"]:
        return "YOLO11n + PConv"
    if module_counts["C3k2_LSKConv"]:
        return "YOLO11n + LSKConv"
    parts = ["YOLO11n"]
    if module_counts["C3k2_InceptionDW"]:
        parts.append("InceptionDW")
    if module_counts["DySample"]:
        parts.append("DPLS")
    elif any(
        module_counts[name]
        for name in (
            "SCAM",
            "CASCAM",
            "CASCAMFixedBeta",
            "CASCAMUnbounded",
            "VGUPPreprocessor",
            "ERUPPreprocessor",
        )
    ):
        parts.append("PLS")
    if module_counts["CASCAM"]:
        parts.append("CA-SCAM")
    elif module_counts["CASCAMFixedBeta"]:
        parts.append("CA-SCAM-FixedBeta")
    elif module_counts["CASCAMUnbounded"]:
        parts.append("CA-SCAM-Unbounded")
    elif module_counts["SCAM"]:
        parts.append("SCAM")
    if module_counts["VGUPPreprocessor"]:
        parts.append("VGUP")
    if module_counts["ERUPPreprocessor"]:
        parts.append("ERUP")
    return " + ".join(parts)


def _best_epoch(
    train_results: dict[str, Any],
    train_metrics: dict[str, Any],
) -> dict[str, Any]:
    epochs = list(train_results.get("epoch", []))
    if not epochs:
        return {"epoch": None, "match_max_abs_diff": None, "method": "unavailable"}
    columns = [column for column in METRIC_KEYS.values() if column in train_results and column in train_metrics]
    if columns:
        candidates = []
        for index in range(len(epochs)):
            diffs = [
                abs(float(train_results[column][index]) - float(train_metrics[column]))
                for column in columns
            ]
            candidates.append((max(diffs), sum(diffs), index))
        maximum, _total, index = min(candidates)
        return {
            "epoch": int(float(epochs[index])),
            "index": index,
            "match_max_abs_diff": maximum,
            "method": "checkpoint train_metrics matched to embedded train_results",
        }
    map_column = METRIC_KEYS["map50_95"]
    if map_column in train_results:
        values = [float(value) for value in train_results[map_column]]
        index = max(range(len(values)), key=values.__getitem__)
        return {
            "epoch": int(float(epochs[index])),
            "index": index,
            "match_max_abs_diff": None,
            "method": "maximum embedded mAP50-95 fallback",
        }
    return {"epoch": None, "match_max_abs_diff": None, "method": "unavailable"}


def _layer_summary(model: torch.nn.Module) -> list[dict[str, Any]]:
    sequence = getattr(model, "model", None)
    if sequence is None:
        return []
    rows = []
    for index, layer in enumerate(sequence):
        rows.append(
            {
                "index": int(getattr(layer, "i", index)),
                "from": _jsonable(getattr(layer, "f", None)),
                "type": type(layer).__name__,
                "parameters": sum(parameter.numel() for parameter in layer.parameters()),
            }
        )
    return rows


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    from custom_modules.register import register_custom_modules

    register_custom_modules(patch_parse_model=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)!r}")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Checkpoint does not contain an EMA/model torch module.")
    return checkpoint, model


def audit_experiment_results(
    results_dir: str | Path,
    repo_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(results_dir)
    root = Path(repo_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    yaml_index = repository_architectures(root)
    csv_payloads = {
        path: _read_csv(path)
        for path in sorted(source.glob("*.csv"))
    }
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in sorted(source.glob("*.pt")):
        try:
            checkpoint, model = _load_checkpoint(path)
            train_args = dict(checkpoint.get("train_args") or getattr(model, "args", {}) or {})
            train_metrics = dict(checkpoint.get("train_metrics") or {})
            train_results = dict(checkpoint.get("train_results") or {})
            yaml_payload = dict(getattr(model, "yaml", {}) or {})
            yaml_digest = architecture_sha256(yaml_payload)
            yaml_matches = yaml_index.get(yaml_digest or "", [])
            module_counts = Counter(type(module).__name__ for module in model.modules())

            comparisons = {
                str(csv_path): compare_training_results(train_results, payload)
                for csv_path, payload in csv_payloads.items()
            }
            comparable = [
                (csv_path, result)
                for csv_path, result in comparisons.items()
                if result.get("comparable")
            ]
            comparable.sort(
                key=lambda pair: (
                    pair[1].get("max_abs_diff", float("inf")),
                    pair[1].get("mean_abs_diff", float("inf")),
                    pair[0],
                )
            )
            matched_csv = comparable[0][0] if comparable else None
            matched_result = comparable[0][1] if comparable else {}
            model_argument = str(train_args.get("model") or "")
            model_basename = Path(model_argument).name if model_argument else ""
            basename_matches = [
                item for item in yaml_matches if Path(item).name == model_basename
            ]
            identity = _human_identity(module_counts)

            if matched_result.get("rounded_exact") and basename_matches:
                status = "已确认"
                evidence = "训练CSV数值匹配 + train_args模型路径匹配 + checkpoint结构匹配"
            elif matched_result.get("rounded_exact") and yaml_matches:
                status = "已确认"
                evidence = "训练CSV数值匹配 + checkpoint结构与仓库YAML匹配"
            elif matched_result.get("rounded_exact"):
                status = "部分确认"
                evidence = "训练CSV数值匹配；仓库中未找到完全相同结构YAML"
            else:
                status = "待人工确认"
                evidence = "未能唯一匹配训练CSV或仓库结构"

            best = _best_epoch(train_results, train_metrics)
            metric_record = {
                name: float(train_metrics[column]) if column in train_metrics else None
                for name, column in METRIC_KEYS.items()
            }
            custom_counts = {
                name: int(module_counts[name])
                for name in sorted(CUSTOM_MODULES)
                if module_counts[name]
            }
            record = {
                "checkpoint": str(path.resolve()),
                "checkpoint_name": path.name,
                "checkpoint_size_bytes": path.stat().st_size,
                "checkpoint_sha256": sha256_file(path),
                "identity": identity,
                "identity_status": status,
                "identity_evidence": evidence,
                "matched_results_csv": matched_csv,
                "csv_match": matched_result,
                "training_version": checkpoint.get("version"),
                "training_date": checkpoint.get("date"),
                "checkpoint_git": _jsonable(checkpoint.get("git")),
                "train_args": _jsonable(train_args),
                "train_metrics": metric_record,
                "train_metrics_raw": _jsonable(train_metrics),
                "best_epoch": best,
                "epochs_recorded": len(train_results.get("epoch", [])),
                "model_argument": model_argument,
                "model_yaml_file": yaml_payload.get("yaml_file"),
                "architecture_sha256": yaml_digest,
                "repository_yaml_matches": yaml_matches,
                "repository_yaml_basename_matches": basename_matches,
                "module_counts": dict(sorted(module_counts.items())),
                "custom_module_counts": custom_counts,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "state_tensors": len(model.state_dict()),
                "detect_strides": [
                    float(value)
                    for value in getattr(model, "stride", torch.tensor([])).detach().cpu()
                ],
                "layer_summary": _layer_summary(model),
            }
            records.append(record)
            del model, checkpoint
        except Exception as error:  # audit must preserve every failure as evidence
            errors.append(
                {
                    "checkpoint": str(path.resolve()),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    report = {
        "results_dir": str(source.resolve()),
        "repo_root": str(root.resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "records": records,
        "errors": errors,
    }
    json_path = output / "权重身份审计.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output / "权重身份审计.csv"
    fields = [
        "checkpoint_name",
        "identity",
        "identity_status",
        "identity_evidence",
        "matched_results_csv",
        "csv_max_abs_diff",
        "best_epoch",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "parameters",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
        "training_version",
        "training_date",
        "model_argument",
        "repository_yaml_matches",
        "custom_module_counts",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record["train_metrics"]
            writer.writerow(
                {
                    "checkpoint_name": record["checkpoint_name"],
                    "identity": record["identity"],
                    "identity_status": record["identity_status"],
                    "identity_evidence": record["identity_evidence"],
                    "matched_results_csv": record["matched_results_csv"],
                    "csv_max_abs_diff": record["csv_match"].get("max_abs_diff"),
                    "best_epoch": record["best_epoch"].get("epoch"),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "map50": metrics["map50"],
                    "map50_95": metrics["map50_95"],
                    "parameters": record["parameters"],
                    "checkpoint_size_bytes": record["checkpoint_size_bytes"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "training_version": record["training_version"],
                    "training_date": record["training_date"],
                    "model_argument": record["model_argument"],
                    "repository_yaml_matches": json.dumps(record["repository_yaml_matches"], ensure_ascii=False),
                    "custom_module_counts": json.dumps(record["custom_module_counts"], ensure_ascii=False),
                }
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = audit_experiment_results(
        args.results_dir,
        args.repo_root,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "identified": len(report["records"]),
                "errors": len(report["errors"]),
                "output": str(Path(args.output_dir).resolve()),
            },
            ensure_ascii=False,
        )
    )
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
