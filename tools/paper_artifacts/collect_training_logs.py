"""Build traceable metric tables and uniform training curves from formal logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRICS = ["precision", "recall", "map50", "map50_95"]
DISPLAY = {
    "precision": "P",
    "recall": "R",
    "map50": "mAP50",
    "map50_95": "mAP50-95",
}
CURVE_COLUMNS = {
    "mAP50-95": "metrics/mAP50-95(B)",
    "mAP50": "metrics/mAP50(B)",
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
    "box loss": "train/box_loss",
    "classification loss": "train/cls_loss",
    "DFL loss": "train/dfl_loss",
}
CURVE_FILENAMES = {
    "mAP50-95": "训练曲线_mAP50-95",
    "mAP50": "训练曲线_mAP50",
    "Precision": "训练曲线_Precision",
    "Recall": "训练曲线_Recall",
    "box loss": "训练曲线_box_loss",
    "classification loss": "训练曲线_classification_loss",
    "DFL loss": "训练曲线_DFL_loss",
}


def _configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or (list(rows[0]) if rows else ["模型"])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _supplemental_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload.get("records", [])


def metric_rows(
    audit: dict[str, Any],
    supplemental: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in audit["records"]:
        metrics = record["train_metrics"]
        rows.append(
            {
                "模型": record["identity"],
                "P": metrics.get("precision"),
                "R": metrics.get("recall"),
                "mAP50": metrics.get("map50"),
                "mAP50-95": metrics.get("map50_95"),
                "best epoch": record.get("best_epoch", {}).get("epoch"),
                "训练epoch数": record.get("epochs_recorded"),
                "权重文件": record["checkpoint_name"],
                "results.csv": Path(record["matched_results_csv"]).name
                if record.get("matched_results_csv")
                else None,
                "权重SHA256": record["checkpoint_sha256"],
                "身份状态": record["identity_status"],
                "数值来源": "best.pt/train_metrics（与嵌入train_results及CSV交叉核对）",
            }
        )
    existing = {row["模型"] for row in rows}
    for item in supplemental:
        if item["模型"] in existing:
            continue
        rows.append(
            {
                "模型": item["模型"],
                "P": item.get("P"),
                "R": item.get("R"),
                "mAP50": item.get("mAP50"),
                "mAP50-95": item.get("mAP50-95"),
                "best epoch": None,
                "训练epoch数": item.get("训练epoch数"),
                "权重文件": None,
                "results.csv": None,
                "权重SHA256": None,
                "身份状态": item.get("身份状态", "待人工确认"),
                "数值来源": item.get("数值来源", "任务文档人工提供；本地未找到对应权重/日志"),
            }
        )
    preferred_order = [
        "YOLO11n",
        "YOLO11n + InceptionDW",
        "YOLO11n + PConv",
        "YOLO11n + LSKConv",
        "YOLO11n + InceptionDW + DPLS",
        "YOLO11n + InceptionDW + DPLS + SCAM",
        "YOLO11n + InceptionDW + DPLS + SCAM + VGUP",
        "YOLO11n + InceptionDW + DPLS + CA-SCAM",
        "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP",
        "YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP",
        "YOLO11n + PLS + CA-SCAM + VGUP",
    ]
    order = {name: index for index, name in enumerate(preferred_order)}
    rows.sort(key=lambda row: (order.get(row["模型"], 10_000), row["模型"]))
    return rows


def _baseline_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["模型"]: row for row in rows}
    baseline = by_name.get("YOLO11n")
    if baseline is None:
        raise ValueError("A confirmed YOLO11n baseline is required for delta tables.")
    output = []
    for row in rows:
        result = dict(row)
        for metric in ("P", "R", "mAP50", "mAP50-95"):
            left, right = row.get(metric), baseline.get(metric)
            result[f"Δ{metric}"] = (
                float(left) - float(right)
                if left is not None and right is not None
                else None
            )
        output.append(result)
    return output


def _neighbour_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["模型"]: row for row in rows}
    comparisons = [
        (
            "CA-SCAM vs SCAM",
            "固定 InceptionDW + DPLS + VGUP",
            "YOLO11n + InceptionDW + DPLS + SCAM + VGUP",
            "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP",
        ),
        (
            "VGUP加入前后",
            "固定 InceptionDW + DPLS + CA-SCAM",
            "YOLO11n + InceptionDW + DPLS + CA-SCAM",
            "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP",
        ),
        (
            "PLS vs DPLS",
            "固定 InceptionDW + CA-SCAM + VGUP",
            "YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP",
            "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP",
        ),
        (
            "InceptionDW加入前后",
            "固定 PLS + CA-SCAM + VGUP",
            "YOLO11n + PLS + CA-SCAM + VGUP",
            "YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP",
        ),
    ]
    output = []
    for comparison, fixed, control_name, treatment_name in comparisons:
        control, treatment = by_name.get(control_name), by_name.get(treatment_name)
        if control is None or treatment is None:
            output.append(
                {
                    "比较": comparison,
                    "固定结构": fixed,
                    "对照模型": control_name,
                    "实验模型": treatment_name,
                    "状态": "缺少对照文件",
                }
            )
            continue
        row = {
            "比较": comparison,
            "固定结构": fixed,
            "对照模型": control_name,
            "实验模型": treatment_name,
            "状态": "可计算",
        }
        for metric in ("P", "R", "mAP50", "mAP50-95"):
            row[f"对照{metric}"] = control[metric]
            row[f"实验{metric}"] = treatment[metric]
            row[f"Δ{metric}"] = float(treatment[metric]) - float(control[metric])
        output.append(row)
    return output


def _convergence_record(label: str, frame: pd.DataFrame) -> dict[str, Any]:
    map_column = CURVE_COLUMNS["mAP50-95"]
    if map_column not in frame or frame.empty:
        return {"模型": label, "状态": "缺少mAP50-95曲线"}
    values = frame[map_column].astype(float).to_numpy()
    epochs = frame["epoch"].astype(float).to_numpy()
    best_index = int(np.nanargmax(values))
    tail_count = min(20, len(values))
    tail = values[-tail_count:]
    x = np.arange(tail_count, dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0]) if tail_count >= 2 else 0.0
    jump_count = int(np.sum(np.abs(np.diff(values)) > 0.05))
    return {
        "模型": label,
        "epoch数": len(values),
        "曲线mAP50-95最高epoch": int(epochs[best_index]),
        "曲线mAP50-95最高值": float(values[best_index]),
        "最后epoch值": float(values[-1]),
        "末20轮标准差": float(np.std(tail)),
        "末20轮线性斜率_每epoch": slope,
        "相邻epoch绝对跳变大于0.05次数": jump_count,
        "best至最后下降": float(values[best_index] - values[-1]),
        "状态": "已分析",
    }


def _plot_curves(
    labelled_frames: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    _configure_matplotlib()
    colors = ["#1F4E79", "#C55A11", "#7F6000", "#6B6B6B", "#7030A0", "#008C95", "#A64D79", "#3D6B35", "#2F5597", "#A61C00"]
    styles = ["-", "--", "-.", ":"]
    for display, column in CURVE_COLUMNS.items():
        available = [(label, frame) for label, frame in labelled_frames if column in frame]
        if not available:
            continue
        figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
        for index, (label, frame) in enumerate(available):
            axis.plot(
                frame["epoch"],
                frame[column],
                color=colors[index % len(colors)],
                linestyle=styles[(index // len(colors)) % len(styles)],
                linewidth=1.45,
                alpha=0.92,
                label=label,
            )
        axis.set_title(f"正式实验训练曲线：{display}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(display)
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            loc="best",
            fontsize=7.2,
            frameon=False,
            ncol=1 if len(available) <= 5 else 2,
        )
        stem = output_dir / CURVE_FILENAMES[display]
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def build_training_materials(
    audit_json: str | Path,
    output_dir: str | Path,
    supplemental_json: str | Path | None = None,
) -> dict[str, Any]:
    audit = json.loads(Path(audit_json).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = metric_rows(audit, _supplemental_rows(supplemental_json))
    deltas = _baseline_delta_rows(rows)
    neighbours = _neighbour_rows(rows)
    _write_csv(output / "01_所有已有模型指标总表.csv", rows)
    _write_csv(output / "03_模型结果变化总表.csv", deltas)
    _write_csv(output / "04_近邻对照差值表.csv", neighbours)

    frames: list[tuple[str, pd.DataFrame]] = []
    convergence = []
    for record in audit["records"]:
        csv_path = record.get("matched_results_csv")
        if not csv_path:
            continue
        frame = pd.read_csv(csv_path)
        frames.append((record["identity"], frame))
        convergence.append(_convergence_record(record["identity"], frame))
    _write_csv(output / "05_训练收敛审计.csv", convergence)
    _plot_curves(frames, output)

    unstable = [
        row["模型"]
        for row in convergence
        if row.get("状态") == "已分析"
        and (
            float(row["末20轮标准差"]) > 0.02
            or int(row["相邻epoch绝对跳变大于0.05次数"]) > 5
        )
    ]
    advice = [
        "# 训练曲线使用建议",
        "",
        "## 建议放入正文",
        "",
        "- mAP50-95 曲线：最直接展示核心评价指标的收敛过程。",
        "- 如版面允许，可补充 mAP50 曲线；建议只保留关键近邻对照模型，避免八条以上曲线拥挤。",
        "",
        "## 建议作为补充材料或审稿备查",
        "",
        "- Precision、Recall 曲线：用于解释 P/R 权衡，不建议与主结果表重复占用正文版面。",
        "- box、classification、DFL loss 曲线：用于证明训练过程正常，通常放补充材料。",
        "",
        "## 客观收敛检查",
        "",
        "收敛审计使用末20轮标准差、末20轮线性斜率、最高值至最后值下降量以及相邻轮次大跳变次数。",
        "具体数值见 `05_训练收敛审计.csv`；该规则只用于发现需要人工复核的曲线，不替代模型优劣判断。",
        "",
        f"- 按预设阈值需要优先人工复核的模型：{('、'.join(unstable) if unstable else '无')}。",
        "- best epoch 与最后 epoch 不相同并不等于异常；正式指标取 best.pt 中保存的 train_metrics。",
        "- 训练曲线来自与 checkpoint 内嵌 train_results 数值匹配的 results.csv，不使用手工抄录曲线。",
        "",
    ]
    (output / "训练曲线使用建议.md").write_text(
        "\n".join(advice),
        encoding="utf-8",
    )
    return {
        "metric_rows": rows,
        "delta_rows": deltas,
        "neighbour_rows": neighbours,
        "convergence": convergence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--supplemental-json")
    args = parser.parse_args()
    report = build_training_materials(
        args.audit_json,
        args.output_dir,
        args.supplemental_json,
    )
    print(
        json.dumps(
            {
                "models": len(report["metric_rows"]),
                "output": str(Path(args.output_dir).resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
