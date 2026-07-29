"""Deterministic quantile, dilution, bin, and split-shift statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import SPLITS, dilution_metrics, quantile_linear

METRICS = ("short_side_640_px", "long_side_640_px", "area_640_px2", "aspect_ratio")
QUANTILE_PROBABILITIES = (0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 1.0)
QUANTILE_COLUMNS = (
    ("Min", 0.0),
    ("Q1", 0.01),
    ("Q2.5", 0.025),
    ("Q5", 0.05),
    ("Q10", 0.10),
    ("Q25", 0.25),
    ("Q50", 0.50),
    ("Q75", 0.75),
    ("Q90", 0.90),
    ("Q95", 0.95),
    ("Q97.5", 0.975),
    ("Q99", 0.99),
    ("Max", 1.0),
)
LOW_QUANTILES = (0.025, 0.05, 0.10)
PYRAMID_STRIDES = (("P2", 4), ("P3", 8), ("P4", 16), ("P5", 32))
BIN_DEFINITIONS = (
    ("[0,4)", 0.0, 4.0),
    ("[4,8)", 4.0, 8.0),
    ("[8,16)", 8.0, 16.0),
    ("[16,32)", 16.0, 32.0),
    ("[32,+inf)", 32.0, np.inf),
)


def split_frame(instances: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return instances
    return instances[instances["split"] == split]


def compute_quantiles(instances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {"method": "linear", "splits": {}}
    probabilities = np.asarray(QUANTILE_PROBABILITIES, dtype=np.float64)
    for split in (*SPLITS, "all"):
        frame = split_frame(instances, split)
        if frame.empty:
            raise ValueError(f"Split {split!r} has no valid instances.")
        nested["splits"][split] = {}
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=np.float64)
            quantiles = quantile_linear(values, probabilities)
            mapping = {float(probability): float(value) for probability, value in zip(probabilities, quantiles, strict=True)}
            nested["splits"][split][metric] = {
                f"{probability:.6g}": mapping[float(probability)] for probability in probabilities
            }
            for probability, value in mapping.items():
                long_rows.append(
                    {
                        "split": split,
                        "metric": metric,
                        "quantile": f"{probability * 100:g}%",
                        "quantile_probability": probability,
                        "value": value,
                    }
                )
            wide = {"Split": split, "Metric": metric}
            for column, probability in QUANTILE_COLUMNS:
                wide[column] = mapping[probability]
            wide_rows.append(wide)
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows), nested


def lookup_quantile(quantiles_long: pd.DataFrame, split: str, metric: str, probability: float) -> float:
    rows = quantiles_long[
        (quantiles_long["split"] == split)
        & (quantiles_long["metric"] == metric)
        & np.isclose(quantiles_long["quantile_probability"].astype(float), probability)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one quantile for {split}/{metric}/{probability}, found {len(rows)}")
    return float(rows.iloc[0]["value"])


def compute_dilution_rates(quantiles_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {"formula": "max(0, 1 - L_short_quantile / stride) * 100", "splits": {}}
    for split in (*SPLITS, "all"):
        nested["splits"][split] = {}
        for probability in LOW_QUANTILES:
            short_side = lookup_quantile(quantiles_long, split, "short_side_640_px", probability)
            q_label = f"{probability * 100:g}%"
            nested["splits"][split][q_label] = {}
            for level, stride in PYRAMID_STRIDES:
                metrics = dilution_metrics(short_side, stride)
                row = {
                    "split": split,
                    "quantile": q_label,
                    "quantile_probability": probability,
                    "short_side_quantile_px": short_side,
                    "pyramid_level": level,
                    "stride": stride,
                    **metrics,
                }
                rows.append(row)
                nested["splits"][split][q_label][level] = {
                    "stride": stride,
                    "short_side_quantile_px": short_side,
                    **metrics,
                }
    long_frame = pd.DataFrame(rows)
    wide_rows: list[dict[str, Any]] = []
    for (split, probability), group in long_frame.groupby(["split", "quantile_probability"], sort=False):
        row: dict[str, Any] = {
            "Split": split,
            "q": f"{probability * 100:g}%",
            "Short-side quantile": float(group.iloc[0]["short_side_quantile_px"]),
        }
        for _, item in group.iterrows():
            level = str(item["pyramid_level"])
            row[f"{level} ratio"] = float(item["sampling_intervals_spanned"])
            row[f"{level} dilution"] = float(item["dilution_rate_percent"])
        wide_rows.append(row)
    return long_frame, pd.DataFrame(wide_rows), nested


def compute_stride_bins(instances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_rows: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    for split in (*SPLITS, "all"):
        values = split_frame(instances, split)["short_side_640_px"].to_numpy(dtype=np.float64)
        total = int(values.size)
        if total == 0:
            raise ValueError(f"Split {split!r} has no valid instances.")
        for label, lower, upper in BIN_DEFINITIONS:
            count = int(np.count_nonzero((values >= lower) & (values < upper)))
            bin_rows.append(
                {
                    "split": split,
                    "short_side_range": label,
                    "lower_bound_inclusive": lower,
                    "upper_bound_exclusive": None if np.isinf(upper) else upper,
                    "count": count,
                    "percentage": count / total * 100.0,
                }
            )
        for threshold in (4, 8, 16, 32):
            count = int(np.count_nonzero(values < threshold))
            cumulative_rows.append(
                {
                    "split": split,
                    "threshold_px": threshold,
                    "count_below": count,
                    "percentage_below": count / total * 100.0,
                }
            )
    return pd.DataFrame(bin_rows), pd.DataFrame(cumulative_rows)


def empirical_ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the two-sample empirical-CDF maximum distance without a p-value."""
    left = np.sort(np.asarray(left, dtype=np.float64))
    right = np.sort(np.asarray(right, dtype=np.float64))
    if left.size == 0 or right.size == 0:
        raise ValueError("Both samples must be non-empty.")
    grid = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, grid, side="right") / left.size
    right_cdf = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def split_shift_statistics(instances: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left_name, right_name in pairs:
        left = split_frame(instances, left_name)["short_side_640_px"].to_numpy(dtype=np.float64)
        right = split_frame(instances, right_name)["short_side_640_px"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "left_split": left_name,
                "right_split": right_name,
                "empirical_cdf_max_distance": empirical_ks_distance(left, right),
                "left_median_px": float(np.median(left)),
                "right_median_px": float(np.median(right)),
            }
        )
    return rows
