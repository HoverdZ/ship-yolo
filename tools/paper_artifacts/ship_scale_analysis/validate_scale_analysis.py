"""Cross-file and formula validation for generated ship-scale artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import SPLITS, dilution_metrics, quantile_linear

REQUIRED_FILES = (
    "raw_tables/ship_instance_scales.csv",
    "raw_tables/dataset_scale_audit.csv",
    "raw_tables/dataset_scale_audit.json",
    "raw_tables/scale_quantiles_long.csv",
    "raw_tables/scale_quantiles_wide.csv",
    "raw_tables/scale_quantiles.json",
    "raw_tables/dilution_rate_by_stride_long.csv",
    "raw_tables/dilution_rate_by_stride_wide.csv",
    "raw_tables/dilution_rate_by_stride.json",
    "raw_tables/short_side_stride_bins.csv",
    "raw_tables/short_side_cumulative_thresholds.csv",
    "paper_tables/paper_table_scale_quantiles.csv",
    "paper_tables/paper_table_dilution_rates.csv",
    "paper_tables/paper_table_stride_bins.csv",
    "reports/ship_scale_analysis_summary.md",
    "reports/ship_scale_analysis_summary.json",
    "manuscript_snippets/manuscript_snippet_dpls_scale_motivation_zh.md",
    "manuscript_snippets/manuscript_snippet_dpls_scale_motivation_en.md",
    "figures/short_side_distribution_train.png",
    "figures/short_side_distribution_train.pdf",
    "figures/short_side_cdf_train.png",
    "figures/short_side_cdf_train.pdf",
    "figures/short_side_distribution_by_split.png",
    "figures/short_side_distribution_by_split.pdf",
    "figures/short_long_joint_distribution_train.png",
    "figures/short_long_joint_distribution_train.pdf",
    "figures/short_long_joint_with_marginals_train.png",
    "figures/short_long_joint_with_marginals_train.pdf",
    "figures/area_distribution_train.png",
    "figures/area_distribution_train.pdf",
    "figures/aspect_ratio_distribution_train.png",
    "figures/aspect_ratio_distribution_train.pdf",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(left: float, right: float, label: str, tolerance: float = 1e-9) -> None:
    if not np.isclose(left, right, rtol=tolerance, atol=tolerance):
        raise AssertionError(f"{label}: {left} != {right}")


def validate_output(output_dir: str | Path, require_checksums: bool = True) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")
    empty = [relative for relative in REQUIRED_FILES if (root / relative).stat().st_size == 0]
    if empty:
        raise AssertionError(f"Empty required artifacts: {empty}")

    instances = pd.read_csv(root / "raw_tables/ship_instance_scales.csv")
    audit = pd.read_csv(root / "raw_tables/dataset_scale_audit.csv")
    quantiles = pd.read_csv(root / "raw_tables/scale_quantiles_long.csv")
    dilution = pd.read_csv(root / "raw_tables/dilution_rate_by_stride_long.csv")
    plot_data = pd.read_csv(root / "raw_tables/plot_short_side_by_split.csv")

    split_total = sum(int((instances["split"] == split).sum()) for split in SPLITS)
    if split_total != len(instances):
        raise AssertionError("train/val/test instance counts do not equal all instances.")
    identifiers = instances[["split", "image_relative_path", "box_index"]].astype(str).agg("|".join, axis=1)
    if identifiers.duplicated().any():
        raise AssertionError("Duplicate instance identifiers were found.")
    if len(plot_data) != len(instances):
        raise AssertionError("Plot-data row count differs from the instance table.")
    for split in SPLITS:
        left = np.sort(instances.loc[instances["split"] == split, "short_side_640_px"].to_numpy(dtype=np.float64))
        right = np.sort(plot_data.loc[plot_data["split"] == split, "short_side_640_px"].to_numpy(dtype=np.float64))
        if not np.allclose(left, right, rtol=1e-11, atol=1e-11):
            raise AssertionError(f"Plot data differs from raw instances for {split}.")
        audit_instances = int(audit.loc[audit["split"] == split, "instance_count"].iloc[0])
        if audit_instances != left.size:
            raise AssertionError(f"Audit instance count differs for {split}.")

    for (split, metric), group in quantiles.groupby(["split", "metric"], sort=False):
        values = instances[metric].to_numpy(dtype=np.float64) if split == "all" else instances.loc[instances["split"] == split, metric].to_numpy(dtype=np.float64)
        expected = quantile_linear(values, group["quantile_probability"].to_numpy(dtype=np.float64))
        if not np.allclose(expected, group["value"].to_numpy(dtype=np.float64), rtol=1e-10, atol=1e-10):
            raise AssertionError(f"Quantile mismatch for {split}/{metric}.")

    for _, row in dilution.iterrows():
        expected = dilution_metrics(float(row["short_side_quantile_px"]), int(row["stride"]))
        _assert_close(
            float(row["sampling_intervals_spanned"]),
            float(expected["sampling_intervals_spanned"]),
            "sampling intervals",
        )
        _assert_close(
            float(row["dilution_rate_percent"]),
            float(expected["dilution_rate_percent"]),
            "dilution rate",
        )
        if not 0.0 <= float(row["dilution_rate_percent"]) <= 100.0:
            raise AssertionError("Dilution rate is outside [0,100].")
        if bool(row["representable_by_one_interval"]) != bool(expected["representable_by_one_interval"]):
            raise AssertionError("Representability boolean mismatch.")

    audit_json = json.loads((root / "raw_tables/dataset_scale_audit.json").read_text(encoding="utf-8"))
    for _, row in audit.iterrows():
        split = str(row["split"])
        if int(row["instance_count"]) != int(audit_json["splits"][split]["instance_count"]):
            raise AssertionError(f"CSV/JSON audit mismatch for {split}.")

    checksum_count = 0
    if require_checksums:
        checksum_file = root / "artifact_checksums.sha256"
        manifest_file = root / "artifact_manifest.json"
        if not checksum_file.is_file() or not manifest_file.is_file():
            raise FileNotFoundError("Manifest/checksum files are missing.")
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            actual = _hash(root / relative)
            if actual != expected:
                raise AssertionError(f"Checksum mismatch: {relative}")
            checksum_count += 1
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for artifact in manifest["artifacts"]:
            path = root / artifact["path"]
            if _hash(path) != artifact["sha256"] or path.stat().st_size != artifact["size_bytes"]:
                raise AssertionError(f"Manifest mismatch: {artifact['path']}")

    sensitive_markers = ("/content/drive/MyDrive/", "C:\\Users\\", "D:\\train", "D:\\val", "D:\\test")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".tex", ".sha256"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for marker in sensitive_markers:
                if marker in text:
                    raise AssertionError(f"Sensitive absolute path marker {marker!r} found in {path.relative_to(root)}")

    return {
        "passed": True,
        "instance_count": int(len(instances)),
        "required_artifact_count": len(REQUIRED_FILES),
        "verified_checksum_count": checksum_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    report = validate_output(args.output_dir, require_checksums=not args.skip_checksums)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
