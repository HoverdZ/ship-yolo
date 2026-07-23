"""Summarize ASCGD Ultralytics run directories into one JSON and CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ascgd_utils import (
    REPORT_DIR,
    VARIANTS,
    model_statistics,
    register_modules,
    write_json,
)


METRIC_COLUMNS = {
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def _clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def _read_best_row(results_csv: Path) -> tuple[dict[str, str], int]:
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [_clean_row(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"No epochs found in {results_csv}.")
    metric = METRIC_COLUMNS["mAP50-95"]
    if metric not in rows[0]:
        raise KeyError(f"Missing {metric!r} in {results_csv}.")
    best = max(rows, key=lambda row: float(row[metric]))
    epoch_key = "epoch"
    return best, int(float(best[epoch_key]))


def _variant_from_run(run_dir: Path, args_yaml: dict[str, Any]) -> str:
    explicit = run_dir / "ascgd_training_request.json"
    if explicit.is_file():
        return json.loads(explicit.read_text(encoding="utf-8"))["variant"]
    name = str(args_yaml.get("name", run_dir.name))
    matches = [
        variant
        for variant, config in VARIANTS.items()
        if config["name"] == name or variant in name
    ]
    return matches[0] if matches else "unknown"


def _model_stats(best_pt: Path) -> dict[str, Any]:
    register_modules()
    from ultralytics import YOLO

    model = YOLO(str(best_pt), verbose=False)
    imgsz = int(getattr(model, "overrides", {}).get("imgsz", 640))
    return model_statistics(model, imgsz=imgsz)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORT_DIR / "results_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPORT_DIR / "results_summary.csv",
    )
    args = parser.parse_args()

    benchmark = {}
    if args.benchmark:
        benchmark_payload = json.loads(
            args.benchmark.expanduser().resolve().read_text(encoding="utf-8")
        )
        benchmark = benchmark_payload.get("variants", {})

    rows = []
    for supplied in args.runs:
        run_dir = supplied.expanduser().resolve()
        results_csv = run_dir / "results.csv"
        args_yaml_path = run_dir / "args.yaml"
        best_pt = run_dir / "weights" / "best.pt"
        missing = [
            str(path)
            for path in (results_csv, args_yaml_path, best_pt)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Run {run_dir} is incomplete; missing {missing}.")
        import yaml

        args_yaml = yaml.safe_load(args_yaml_path.read_text(encoding="utf-8"))
        best, best_epoch = _read_best_row(results_csv)
        variant = _variant_from_run(run_dir, args_yaml)
        stats = _model_stats(best_pt)
        amp_benchmark = benchmark.get(variant, {}).get("amp_fp16", {})
        row = {
            "variant": variant,
            "Precision": float(best[METRIC_COLUMNS["Precision"]]),
            "Recall": float(best[METRIC_COLUMNS["Recall"]]),
            "mAP50": float(best[METRIC_COLUMNS["mAP50"]]),
            "mAP50-95": float(best[METRIC_COLUMNS["mAP50-95"]]),
            "best_epoch": best_epoch,
            "parameters": stats["parameters"],
            "GFLOPs": stats["gflops"],
            "FP16_latency": amp_benchmark.get("mean_latency_ms"),
            "peak_memory": amp_benchmark.get("peak_memory_mib"),
            "run_dir": str(run_dir),
            "best_pt": str(best_pt),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    write_json(args.output_json, {"runs": rows})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
        "best_epoch",
        "parameters",
        "GFLOPs",
        "FP16_latency",
        "peak_memory",
        "run_dir",
        "best_pt",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_json.resolve()} and {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
