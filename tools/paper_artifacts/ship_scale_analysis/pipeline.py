"""End-to-end deterministic ship-scale analysis pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .core import SPLITS, analyze_dataset
from .io_utils import (
    build_manifest,
    deterministic_zip,
    stable_environment,
    write_checksums,
    write_csv,
    write_json,
)
from .plots import generate_all_figures
from .reporting import build_summary, generate_paper_tables, write_manuscript_snippets, write_summary
from .statistics import compute_dilution_rates, compute_quantiles, compute_stride_bins
from .validate_scale_analysis import validate_output


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    resolved = output_dir.expanduser().resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ValueError(f"Refusing unsafe output directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already contains artifacts: {resolved}. Use --overwrite explicitly.")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _audit_json(audit: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for _, row in audit.iterrows():
        values = {}
        for key, value in row.items():
            if key == "split":
                continue
            if pd.isna(value):
                values[key] = None
            elif hasattr(value, "item"):
                values[key] = value.item()
            else:
                values[key] = value
        splits[str(row["split"])] = values
    return {"metadata": metadata, "splits": splits}


def _write_plot_data(instances: pd.DataFrame, raw_dir: Path) -> None:
    columns = ["split", "short_side_640_px", "long_side_640_px", "area_640_px2", "aspect_ratio"]
    write_csv(instances[columns], raw_dir / "plot_short_side_by_split.csv")
    train = instances[instances["split"] == "train"]
    write_csv(train[["short_side_640_px"]], raw_dir / "plot_short_side_train.csv")
    write_csv(train[["short_side_640_px", "long_side_640_px"]], raw_dir / "plot_short_long_train.csv")
    write_csv(train[["area_640_px2", "aspect_ratio"]], raw_dir / "plot_area_aspect_train.csv")


def _write_parquet_if_available(instances: pd.DataFrame, raw_dir: Path) -> dict[str, Any]:
    destination = raw_dir / "ship_instance_scales.parquet"
    try:
        instances.to_parquet(destination, index=False)
        return {"available": True, "path": destination.name}
    except (ImportError, ModuleNotFoundError) as error:
        report = {
            "available": False,
            "reason": f"{type(error).__name__}: install pyarrow or fastparquet to enable Parquet output",
        }
        write_json(report, raw_dir / "parquet_unavailable.json")
        return report


def _publish_versioned(output_dir: Path, publish_root: Path, fingerprint: str) -> Path:
    target_root = publish_root.expanduser().resolve() / "数据集尺度分析"
    target_root.mkdir(parents=True, exist_ok=True)
    prefix = f"ship_scale_imgsz640_{fingerprint[:12]}_v"
    version = 1
    while (target_root / f"{prefix}{version}").exists():
        version += 1
    destination = target_root / f"{prefix}{version}"
    shutil.copytree(output_dir, destination)
    return destination


def run_analysis(
    data_yaml: str | Path,
    output_dir: str | Path,
    *,
    imgsz: int = 640,
    dataset_root: str | Path | None = None,
    source_label: str = "ship_detection/data/data.yaml",
    overwrite: bool = False,
    publish_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate the complete artifact set without training, validation, or source writes."""
    if imgsz != 640:
        raise ValueError("This paper artifact schema is fixed to TARGET_IMGSZ=640.")
    output = Path(output_dir).expanduser().resolve()
    _prepare_output(output, overwrite=overwrite)
    raw_dir = output / "raw_tables"
    paper_dir = output / "paper_tables"
    figure_dir = output / "figures"
    report_dir = output / "reports"
    snippet_dir = output / "manuscript_snippets"
    checksum_dir = output / "checksums"
    for directory in (raw_dir, paper_dir, figure_dir, report_dir, snippet_dir, checksum_dir):
        directory.mkdir(parents=True, exist_ok=True)

    instances, audit, issues, metadata = analyze_dataset(
        data_yaml,
        imgsz=imgsz,
        dataset_root=dataset_root,
        source_label=source_label,
    )
    if instances.empty:
        raise ValueError("No valid ship instances were found.")
    write_csv(instances, raw_dir / "ship_instance_scales.csv")
    parquet = _write_parquet_if_available(instances, raw_dir)
    write_csv(audit, raw_dir / "dataset_scale_audit.csv")
    write_json(_audit_json(audit, metadata), raw_dir / "dataset_scale_audit.json")
    write_csv(issues, raw_dir / "reported_source_issues.csv")

    quantiles_long, quantiles_wide, quantiles_json = compute_quantiles(instances)
    write_csv(quantiles_long, raw_dir / "scale_quantiles_long.csv")
    write_csv(quantiles_wide, raw_dir / "scale_quantiles_wide.csv")
    write_json(quantiles_json, raw_dir / "scale_quantiles.json")

    dilution_long, dilution_wide, dilution_json = compute_dilution_rates(quantiles_long)
    write_csv(dilution_long, raw_dir / "dilution_rate_by_stride_long.csv")
    write_csv(dilution_wide, raw_dir / "dilution_rate_by_stride_wide.csv")
    write_json(dilution_json, raw_dir / "dilution_rate_by_stride.json")

    bins, cumulative = compute_stride_bins(instances)
    write_csv(bins, raw_dir / "short_side_stride_bins.csv")
    write_csv(cumulative, raw_dir / "short_side_cumulative_thresholds.csv")
    _write_plot_data(instances, raw_dir)

    generate_all_figures(instances, quantiles_long, figure_dir)
    generate_paper_tables(quantiles_long, dilution_long, bins, paper_dir)
    summary = build_summary(instances, audit, quantiles_long, dilution_long, cumulative, metadata)
    write_summary(summary, report_dir)
    write_manuscript_snippets(summary, snippet_dir)

    generation_report = {
        "source_label": source_label,
        "source_is_read_only": True,
        "source_files_modified": False,
        "training_started": False,
        "yolo_validation_started": False,
        "imgsz": imgsz,
        "splits": list(SPLITS),
        "quantile_method": "numpy.quantile(method='linear')",
        "parquet": parquet,
        "determinism": stable_environment(),
    }
    write_json(generation_report, report_dir / "generation_report.json")
    write_json({"status": "pending final checksum validation"}, checksum_dir / "validation_report.json")

    manifest = build_manifest(output, metadata | {"parquet": parquet})
    write_json(manifest, output / "artifact_manifest.json")
    write_checksums(output)
    validation = validate_output(output, require_checksums=True)
    write_json(validation, checksum_dir / "validation_report.json")
    manifest = build_manifest(output, metadata | {"parquet": parquet})
    write_json(manifest, output / "artifact_manifest.json")
    write_checksums(output)
    validation = validate_output(output, require_checksums=True)

    published_to = None
    bundle = None
    if publish_root is not None:
        published_to = _publish_versioned(output, Path(publish_root), metadata["dataset_fingerprint_sha256"])
    else:
        bundle = deterministic_zip(output, output.parent / "ship_scale_analysis_bundle.zip")
    return {
        "output_dir": str(output),
        "published_to": str(published_to) if published_to else None,
        "bundle": str(bundle) if bundle else None,
        "dataset_fingerprint_sha256": metadata["dataset_fingerprint_sha256"],
        "validation": validation,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--dataset-root", default=None, help="Optional read-only local mirror root.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="analysis/ship_scale")
    parser.add_argument("--source-label", default="ship_detection/data/data.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--publish-root", default=None, help="Optional paper root; a versioned 数据集尺度分析 subfolder is created.")
    args = parser.parse_args()
    result = run_analysis(
        data_yaml=args.data_yaml,
        dataset_root=args.dataset_root,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        source_label=args.source_label,
        overwrite=args.overwrite,
        publish_root=args.publish_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
