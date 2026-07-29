"""Publication-ready Matplotlib figures generated from saved scale statistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from .core import SPLITS, quantile_linear
from .statistics import lookup_quantile

COLORS = {"train": "#1f4e79", "val": "#c55a11", "test": "#548235"}
STRIDES = (("P2", 4), ("P3", 8), ("P4", 16), ("P5", 32))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "ship-yolo deterministic ship-scale analysis"},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "ship-yolo deterministic ship-scale analysis",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return [png, pdf]


def _draw_stride_lines(axis: plt.Axes) -> None:
    for index, (level, stride) in enumerate(STRIDES):
        axis.axvline(stride, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8, alpha=0.85)
        axis.text(stride, 0.98 - index * 0.055, level, transform=axis.get_xaxis_transform(), ha="center", va="top", color="#555555", fontsize=7)


def _display_limit(values: np.ndarray, minimum: float = 36.0) -> float:
    return max(minimum, float(quantile_linear(values, [0.995])[0]) * 1.05)


def plot_short_side_distribution(
    train: pd.DataFrame,
    quantiles_long: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    configure_matplotlib()
    values = train["short_side_640_px"].to_numpy(dtype=np.float64)
    display_max = _display_limit(values)
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.hist(values, bins=60, range=(0, display_max), color=COLORS["train"], alpha=0.78, edgecolor="white", linewidth=0.3)
    _draw_stride_lines(axis)
    q5 = lookup_quantile(quantiles_long, "train", "short_side_640_px", 0.05)
    axis.axvline(q5, color="#9c2f3e", linewidth=1.5, label=f"Train Q5={q5:.2f} px")
    axis.set(xlabel="Short side at 640-pixel input / px", ylabel="Instance count", title="Training-set short-side distribution")
    axis.set_xlim(0, display_max)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper right", frameon=False)
    fig.subplots_adjust(bottom=0.19)
    fig.text(0.01, 0.025, f"Display truncated at the 99.5th percentile ({display_max / 1.05:.2f} px); source CSV retains all instances.", fontsize=7, color="#555555")
    return _save(fig, output_dir, "short_side_distribution_train")


def plot_short_side_cdf(
    train: pd.DataFrame,
    quantiles_long: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    configure_matplotlib()
    values = np.sort(train["short_side_640_px"].to_numpy(dtype=np.float64))
    y = np.arange(1, values.size + 1, dtype=np.float64) / values.size
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.step(values, y, where="post", color=COLORS["train"], linewidth=1.5, label="Train ECDF")
    _draw_stride_lines(axis)
    for probability, color in ((0.025, "#8064a2"), (0.05, "#9c2f3e"), (0.10, "#00a6a6")):
        value = lookup_quantile(quantiles_long, "train", "short_side_640_px", probability)
        axis.axvline(value, color=color, linewidth=1.0, alpha=0.9, label=f"Q{probability * 100:g}={value:.2f} px")
    axis.set(xlabel="Short side at 640-pixel input / px", ylabel="Cumulative proportion", title="Training-set short-side empirical CDF")
    axis.set_ylim(0, 1.005)
    axis.set_xlim(0, _display_limit(values))
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", frameon=False, ncol=2)
    return _save(fig, output_dir, "short_side_cdf_train")


def plot_split_comparison(instances: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    all_values = instances["short_side_640_px"].to_numpy(dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    for split in SPLITS:
        values = np.sort(instances.loc[instances["split"] == split, "short_side_640_px"].to_numpy(dtype=np.float64))
        y = np.arange(1, values.size + 1, dtype=np.float64) / values.size
        axis.step(values, y, where="post", color=COLORS[split], linewidth=1.35, label=f"{split} (n={values.size})")
    _draw_stride_lines(axis)
    axis.set(
        xlabel="Short side at 640-pixel input / px",
        ylabel="Cumulative proportion",
        title="Short-side distribution by frozen split",
    )
    axis.set_xlim(0, _display_limit(all_values))
    axis.set_ylim(0, 1.005)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    return _save(fig, output_dir, "short_side_distribution_by_split")


def _joint_axis(axis: plt.Axes, short: np.ndarray, long: np.ndarray, q5: float) -> None:
    x_max = _display_limit(short)
    y_max = max(48.0, float(quantile_linear(long, [0.995])[0]) * 1.05)
    plot = axis.hexbin(short, long, gridsize=48, mincnt=1, bins="log", cmap="viridis", linewidths=0, extent=(0, x_max, 0, y_max))
    _draw_stride_lines(axis)
    axis.axvline(q5, color="#9c2f3e", linewidth=1.4, label=f"Train Q5={q5:.2f} px")
    x = np.linspace(0, x_max, 256)
    for ratio, style in ((2, "--"), (4, "-."), (8, ":")):
        axis.plot(x, ratio * x, color="#777777", linestyle=style, linewidth=0.7, alpha=0.75, label=f"{ratio}:1")
    axis.set_xlim(0, x_max)
    axis.set_ylim(0, y_max)
    axis.set_xlabel("Short side at 640-pixel input / px")
    axis.set_ylabel("Long side at 640-pixel input / px")
    axis.legend(frameon=True, facecolor="white", framealpha=0.88, edgecolor="none", loc="lower right", ncol=2)
    return plot


def plot_short_long_joint(train: pd.DataFrame, quantiles_long: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    short = train["short_side_640_px"].to_numpy(dtype=np.float64)
    long = train["long_side_640_px"].to_numpy(dtype=np.float64)
    q5 = lookup_quantile(quantiles_long, "train", "short_side_640_px", 0.05)
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    plot = _joint_axis(axis, short, long, q5)
    axis.set_title("Training-set short-side–long-side joint distribution")
    colorbar = fig.colorbar(plot, ax=axis, pad=0.02)
    colorbar.set_label("log10(count)")
    fig.subplots_adjust(bottom=0.14)
    fig.text(0.01, 0.02, "Axes display the 99.5th-percentile range; the source CSV retains the full, unfiltered distribution.", fontsize=7, color="#555555")
    return _save(fig, output_dir, "short_long_joint_distribution_train")


def plot_short_long_marginals(train: pd.DataFrame, quantiles_long: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    short = train["short_side_640_px"].to_numpy(dtype=np.float64)
    long = train["long_side_640_px"].to_numpy(dtype=np.float64)
    q5 = lookup_quantile(quantiles_long, "train", "short_side_640_px", 0.05)
    fig = plt.figure(figsize=(7.4, 6.2))
    grid = GridSpec(4, 4, figure=fig, hspace=0.08, wspace=0.08)
    top = fig.add_subplot(grid[0, :3])
    right = fig.add_subplot(grid[1:, 3])
    joint = fig.add_subplot(grid[1:, :3])
    plot = _joint_axis(joint, short, long, q5)
    top.hist(short, bins=60, color=COLORS["train"], alpha=0.78)
    top.set_xlim(joint.get_xlim())
    top.set_ylabel("Count")
    top.tick_params(labelbottom=False)
    right.hist(long, bins=60, orientation="horizontal", color="#70ad47", alpha=0.78)
    right.set_ylim(joint.get_ylim())
    right.set_xlabel("Count")
    right.tick_params(labelleft=False)
    colorbar = fig.colorbar(plot, ax=[joint, top, right], pad=0.02, fraction=0.035)
    colorbar.set_label("log10(count)")
    fig.suptitle("Training-set joint scale distribution with marginals", y=0.99)
    return _save(fig, output_dir, "short_long_joint_with_marginals_train")


def plot_area_distribution(train: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    values = train["area_640_px2"].to_numpy(dtype=np.float64)
    display_max = float(quantile_linear(values, [0.995])[0]) * 1.05
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.hist(values, bins=70, range=(0, display_max), color="#4472c4", alpha=0.8, edgecolor="white", linewidth=0.25)
    axis.set(xlabel="Box area at 640-pixel input / px²", ylabel="Instance count", title="Training-set box-area distribution")
    axis.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(bottom=0.19)
    fig.text(0.01, 0.025, "Display truncated at the 99.5th percentile; the source CSV retains all instances.", fontsize=7, color="#555555")
    return _save(fig, output_dir, "area_distribution_train")


def plot_aspect_ratio_distribution(train: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    values = train["aspect_ratio"].to_numpy(dtype=np.float64)
    display_max = max(4.2, float(quantile_linear(values, [0.995])[0]) * 1.05)
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.hist(values, bins=60, range=(1, display_max), color="#ed7d31", alpha=0.8, edgecolor="white", linewidth=0.25)
    for ratio in (2, 4):
        if ratio <= display_max:
            axis.axvline(ratio, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8)
    axis.set(xlabel="Horizontal-box aspect ratio (long/short)", ylabel="Instance count", title="Training-set aspect-ratio distribution")
    axis.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(bottom=0.19)
    fig.text(0.01, 0.025, "Display truncated at the 99.5th percentile; the source CSV retains all instances.", fontsize=7, color="#555555")
    return _save(fig, output_dir, "aspect_ratio_distribution_train")


def generate_all_figures(
    instances: pd.DataFrame,
    quantiles_long: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    train = instances[instances["split"] == "train"].copy()
    created: list[Path] = []
    created.extend(plot_short_side_distribution(train, quantiles_long, output_dir))
    created.extend(plot_short_side_cdf(train, quantiles_long, output_dir))
    created.extend(plot_split_comparison(instances, output_dir))
    created.extend(plot_short_long_joint(train, quantiles_long, output_dir))
    created.extend(plot_short_long_marginals(train, quantiles_long, output_dir))
    created.extend(plot_area_distribution(train, output_dir))
    created.extend(plot_aspect_ratio_distribution(train, output_dir))
    return created
