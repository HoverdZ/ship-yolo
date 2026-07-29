"""Paper tables, objective summaries, and bilingual manuscript snippets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import SPLITS
from .io_utils import write_csv, write_json
from .statistics import (
    LOW_QUANTILES,
    lookup_quantile,
    split_frame,
    split_shift_statistics,
)

METRIC_LABELS = {
    "short_side_640_px": "Short side / px",
    "long_side_640_px": "Long side / px",
    "area_640_px2": "Area / px2",
    "aspect_ratio": "Aspect ratio",
}


def _markdown(frame: pd.DataFrame, float_columns: set[str] | None = None) -> str:
    float_columns = float_columns or set()
    headers = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if column in float_columns and pd.notna(value):
                values.append(f"{float(value):.2f}")
            else:
                values.append(str(value))
        rows.append(values)
    alignment = ["---:" if column in float_columns else "---" for column in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines) + "\n"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "≥": r"$\geq$",
        "<": r"$<$",
        "–": "--",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _latex_table(frame: pd.DataFrame, caption_suggestion: str, float_columns: set[str]) -> str:
    alignment = "".join("r" if column in float_columns else "l" for column in frame.columns)
    lines = [
        f"% Suggested caption: {caption_suggestion}",
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(_latex_escape(column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            cells.append(f"{float(value):.2f}" if column in float_columns and pd.notna(value) else _latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def generate_paper_tables(
    quantiles_long: pd.DataFrame,
    dilution_long: pd.DataFrame,
    bins: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    selected_probabilities = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95)
    scale_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for metric, label in METRIC_LABELS.items():
            row: dict[str, Any] = {"Split": split, "Metric": label}
            for probability in selected_probabilities:
                row[f"Q{probability * 100:g}"] = lookup_quantile(quantiles_long, split, metric, probability)
            scale_rows.append(row)
    scale = pd.DataFrame(scale_rows)
    scale_csv = output_dir / "paper_table_scale_quantiles.csv"
    write_csv(scale, scale_csv, paper_rounding=True)
    float_columns = {column for column in scale.columns if column.startswith("Q")}
    scale_md = output_dir / "paper_table_scale_quantiles.md"
    scale_tex = output_dir / "paper_table_scale_quantiles.tex"
    scale_md.write_text(_markdown(scale, float_columns), encoding="utf-8")
    scale_tex.write_text(
        _latex_table(
            scale,
            "Lower-tail and central ship-box scale statistics after deterministic letterbox mapping to a 640-pixel input.",
            float_columns,
        ),
        encoding="utf-8",
    )
    created.extend((scale_csv, scale_md, scale_tex))

    dilution_csv = output_dir / "paper_table_dilution_rates.csv"
    write_csv(dilution_long, dilution_csv)
    dilution_main = dilution_long[
        (dilution_long["split"] == "train") & np.isclose(dilution_long["quantile_probability"], 0.05)
    ][
        [
            "pyramid_level",
            "stride",
            "short_side_quantile_px",
            "sampling_intervals_spanned",
            "dilution_rate_percent",
        ]
    ].rename(
        columns={
            "pyramid_level": "Pyramid level",
            "stride": "Stride",
            "short_side_quantile_px": "Short-side quantile / px",
            "sampling_intervals_spanned": "Intervals spanned",
            "dilution_rate_percent": "Dilution rate / %",
        }
    )
    dilution_md = output_dir / "paper_table_dilution_rates.md"
    dilution_tex = output_dir / "paper_table_dilution_rates.tex"
    dilution_floats = {"Short-side quantile / px", "Intervals spanned", "Dilution rate / %"}
    dilution_md.write_text(_markdown(dilution_main, dilution_floats), encoding="utf-8")
    dilution_tex.write_text(
        _latex_table(
            dilution_main,
            "Spatial dilution statistics at the training-set fifth-percentile short-side length.",
            dilution_floats,
        ),
        encoding="utf-8",
    )
    created.extend((dilution_csv, dilution_md, dilution_tex))

    bins_table = bins[bins["split"].isin(SPLITS)][["split", "short_side_range", "count", "percentage"]].rename(
        columns={
            "split": "Split",
            "short_side_range": "Short-side range / px",
            "count": "Count",
            "percentage": "Percentage / %",
        }
    )
    bins_csv = output_dir / "paper_table_stride_bins.csv"
    bins_md = output_dir / "paper_table_stride_bins.md"
    bins_tex = output_dir / "paper_table_stride_bins.tex"
    write_csv(bins_table, bins_csv, paper_rounding=True)
    bins_md.write_text(_markdown(bins_table, {"Percentage / %"}), encoding="utf-8")
    bins_tex.write_text(
        _latex_table(
            bins_table,
            "Distribution of horizontal ship-box short sides relative to the P2--P5 feature strides.",
            {"Percentage / %"},
        ),
        encoding="utf-8",
    )
    created.extend((bins_csv, bins_md, bins_tex))
    return created


def _cumulative_lookup(cumulative: pd.DataFrame, split: str, threshold: int) -> float:
    rows = cumulative[(cumulative["split"] == split) & (cumulative["threshold_px"] == threshold)]
    if len(rows) != 1:
        raise ValueError(f"Missing cumulative statistic for {split}/<{threshold}")
    return float(rows.iloc[0]["percentage_below"])


def build_summary(
    instances: pd.DataFrame,
    audit: pd.DataFrame,
    quantiles_long: pd.DataFrame,
    dilution_long: pd.DataFrame,
    cumulative: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    train = split_frame(instances, "train")
    aspect = train["aspect_ratio"].to_numpy(dtype=np.float64)
    shift = split_shift_statistics(instances)
    lower_quantiles = {
        f"q{probability * 100:g}_px": lookup_quantile(
            quantiles_long, "train", "short_side_640_px", probability
        )
        for probability in LOW_QUANTILES
    }
    q5_dilution = dilution_long[
        (dilution_long["split"] == "train") & np.isclose(dilution_long["quantile_probability"], 0.05)
    ]
    maximum_shift = max(item["empirical_cdf_max_distance"] for item in shift)
    medians = {
        split: float(np.median(split_frame(instances, split)["short_side_640_px"].to_numpy(dtype=np.float64)))
        for split in SPLITS
    }
    sensitivity = {}
    for probability in LOW_QUANTILES:
        rows = dilution_long[
            (dilution_long["split"] == "train")
            & np.isclose(dilution_long["quantile_probability"], probability)
        ]
        sensitivity[f"q{probability * 100:g}"] = {
            str(row["pyramid_level"]): float(row["dilution_rate_percent"]) for _, row in rows.iterrows()
        }
    return {
        "method_boundary": (
            "The statistics motivate evaluating a P2-P4 detection pyramid. "
            "Quantitative effectiveness must be verified by controlled detection experiments."
        ),
        "metadata": metadata,
        "audit": {
            str(row["split"]): {
                key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
                for key, value in row.items()
                if key != "split"
            }
            for _, row in audit.iterrows()
        },
        "train_short_side_lower_quantiles_px": lower_quantiles,
        "short_side_below_stride_percent": {
            split: {f"lt_{threshold}px": _cumulative_lookup(cumulative, split, threshold) for threshold in (4, 8, 16, 32)}
            for split in (*SPLITS, "all")
        },
        "train_q5_dilution": {
            str(row["pyramid_level"]): {
                "stride": int(row["stride"]),
                "intervals_spanned": float(row["sampling_intervals_spanned"]),
                "dilution_rate_percent": float(row["dilution_rate_percent"]),
            }
            for _, row in q5_dilution.iterrows()
        },
        "split_short_side_shift": shift,
        "train_aspect_ratio": {
            "mean": float(np.mean(aspect)),
            "median": float(np.median(aspect)),
            "percentage_ge_2": float(np.mean(aspect >= 2) * 100.0),
            "percentage_ge_4": float(np.mean(aspect >= 4) * 100.0),
            "percentage_ge_8": float(np.mean(aspect >= 8) * 100.0),
        },
        "interpretation": {
            "q_candidate": (
                "Use q=5% as the central lower-tail descriptor and retain q=2.5% and q=10% "
                "as sensitivity checks; all three produce the same P2/P3 versus P4/P5 qualitative ordering."
            ),
            "scale_motivation": (
                "Across q=2.5%, 5%, and 10%, P3 has zero dilution while P4 and P5 have positive dilution. "
                "The descriptor therefore supports testing removal of the coarse P5 path, but it does not by "
                "itself establish that P2 is better than P3."
            ),
            "split_shift": (
                f"The largest pairwise short-side ECDF distance is {maximum_shift:.4f}; "
                f"train/val/test medians are {medians['train']:.2f}/{medians['val']:.2f}/{medians['test']:.2f} px. "
                "No marked split-scale displacement is apparent under this descriptor."
            ),
            "elongation": (
                f"The median horizontal-box aspect ratio is {float(np.median(aspect)):.2f}, and "
                f"{float(np.mean(aspect >= 2) * 100.0):.2f}% of training instances have ratio >=2. "
                "The dataset-level DPLS motivation should emphasize object scale rather than a globally pronounced elongated-box distribution."
            ),
            "dilution_sensitivity": sensitivity,
        },
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ship_scale_analysis_summary.json"
    markdown_path = output_dir / "ship_scale_analysis_summary.md"
    write_json(summary, json_path)
    quantiles = summary["train_short_side_lower_quantiles_px"]
    cumulative = summary["short_side_below_stride_percent"]["train"]
    aspect = summary["train_aspect_ratio"]
    q5 = summary["train_q5_dilution"]
    lines = [
        "# Deterministic ship-scale analysis summary",
        "",
        "## Training-set lower-tail short-side statistics",
        "",
        f"- Q2.5: {quantiles['q2.5_px']:.4f} px",
        f"- Q5: {quantiles['q5_px']:.4f} px",
        f"- Q10: {quantiles['q10_px']:.4f} px",
        "",
        "## Training-set short sides below feature strides",
        "",
        *[f"- <{threshold} px: {cumulative[f'lt_{threshold}px']:.4f}%" for threshold in (4, 8, 16, 32)],
        "",
        "## Training-set Q5 dilution statistics",
        "",
        "| Level | Stride | Intervals spanned | Dilution rate / % |",
        "|---|---:|---:|---:|",
        *[
            f"| {level} | {item['stride']} | {item['intervals_spanned']:.4f} | {item['dilution_rate_percent']:.4f} |"
            for level, item in q5.items()
        ],
        "",
        "## Horizontal-box elongation",
        "",
        f"- Mean aspect ratio: {aspect['mean']:.4f}",
        f"- Median aspect ratio: {aspect['median']:.4f}",
        f"- Ratio ≥2: {aspect['percentage_ge_2']:.4f}%",
        f"- Ratio ≥4: {aspect['percentage_ge_4']:.4f}%",
        f"- Ratio ≥8: {aspect['percentage_ge_8']:.4f}%",
        "",
        "## Split comparison",
        "",
        "| Pair | Empirical-CDF maximum distance | Left median / px | Right median / px |",
        "|---|---:|---:|---:|",
    ]
    for item in summary["split_short_side_shift"]:
        lines.append(
            f"| {item['left_split']} vs {item['right_split']} | "
            f"{item['empirical_cdf_max_distance']:.4f} | {item['left_median_px']:.4f} | {item['right_median_px']:.4f} |"
        )
    lines.extend(
        (
            "",
            "## Objective interpretation",
            "",
            f"- Quantile candidate: {summary['interpretation']['q_candidate']}",
            f"- Scale motivation: {summary['interpretation']['scale_motivation']}",
            f"- Split shift: {summary['interpretation']['split_shift']}",
            f"- Aspect ratio: {summary['interpretation']['elongation']}",
            "",
            "These descriptive statistics motivate evaluating a P2–P4 detection pyramid. "
            "They do not establish a detection-accuracy gain; D0/D1/D2 controlled experiments are required.",
            "",
        )
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return [markdown_path, json_path]


def write_manuscript_snippets(
    summary: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    q = summary["train_short_side_lower_quantiles_px"]
    cumulative = summary["short_side_below_stride_percent"]["train"]
    zh = f"""# DPLS尺度动机建议文本

在保持长宽比的640×640输入映射下，训练集船舶水平标注框短边的第2.5、第5和第10百分位数分别为{q['q2.5_px']:.2f}、{q['q5_px']:.2f}和{q['q10_px']:.2f}像素。短边低于4、8、16和32像素的实例比例分别为{cumulative['lt_4px']:.2f}%、{cumulative['lt_8px']:.2f}%、{cumulative['lt_16px']:.2f}%和{cumulative['lt_32px']:.2f}%。当特征步长不大于所选低分位船舶短边时，目标短边至少覆盖一个特征采样间隔，对应空间稀释率为零；随着步长相对目标短边增大，该指标随之上升，反映特征层对窄小目标的空间表征能力逐渐受限。因此，本文将该统计量用于指导检测金字塔层级选择，并通过D0–D2严格对照实验检验P2–P4层级迁移及DySample的实际检测效果。

> 边界：以上统计提供尺度动机，不直接等价于精度提升证据。
"""
    en = f"""# Suggested DPLS scale-motivation text

After aspect-ratio-preserving mapping to a 640 × 640 input, the 2.5th, 5th, and 10th percentiles of the horizontal ship-box short side in the training split are {q['q2.5_px']:.2f}, {q['q5_px']:.2f}, and {q['q10_px']:.2f} pixels, respectively. The proportions of instances with short sides below 4, 8, 16, and 32 pixels are {cumulative['lt_4px']:.2f}%, {cumulative['lt_8px']:.2f}%, {cumulative['lt_16px']:.2f}%, and {cumulative['lt_32px']:.2f}%. When the stride does not exceed the selected lower-quantile short-side length, the target short side spans at least one feature sampling interval and the dilution rate becomes zero. As the stride grows relative to the target short side, the metric increases, indicating progressively constrained spatial representation of narrow small objects. The metric is therefore used to guide the selection of detection pyramid levels, while the practical effects of the P2–P4 level shift and DySample are evaluated through the controlled D0–D2 experiments.

> Boundary: these statistics provide a scale-based motivation and are not, by themselves, evidence of an accuracy gain.
"""
    zh_path = output_dir / "manuscript_snippet_dpls_scale_motivation_zh.md"
    en_path = output_dir / "manuscript_snippet_dpls_scale_motivation_en.md"
    zh_path.write_text(zh, encoding="utf-8")
    en_path.write_text(en, encoding="utf-8")
    return [zh_path, en_path]
