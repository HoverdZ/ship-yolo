"""Create formal ablation tables and clean cumulative comparison charts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_artifacts.formal_protocol import EXPERIMENTS


def summarize(root: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    source = Path(root)
    output = Path(output_dir) if output_dir else source / "paper_summary"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment_id, spec in EXPERIMENTS.items():
        run = source / experiment_id
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = manifest["best_metrics"]
        complexity = json.loads((run / "complexity.json").read_text(encoding="utf-8"))
        best = json.loads((run / "best_epoch_summary.json").read_text(encoding="utf-8"))
        rows.append({
            "Experiment": experiment_id,
            "Model": spec["model_name"],
            "P": metrics["precision"],
            "R": metrics["recall"],
            "mAP50": metrics["map50"],
            "mAP50-95": metrics["map50_95"],
            "AP75": metrics.get("map75"),
            "Params": complexity["parameters"],
            "GFLOPs": complexity["gflops"],
            "Model Size": complexity["model_size_bytes"],
            "Latency": complexity["pytorch_fp32"]["mean_ms"],
            "Best Epoch": best["best_epoch"],
            "Training Time": best["training_time_seconds"],
        })
    frame = pd.DataFrame(rows)
    baseline = frame.iloc[0]
    for metric in ("P", "R", "mAP50", "mAP50-95", "Params", "GFLOPs"):
        frame[f"Δ{metric}"] = frame[metric] - baseline[metric]
    frame.to_csv(output / "formal_ablation_results.csv", index=False)
    try:
        frame.to_excel(output / "formal_ablation_results.xlsx", index=False)
    except ImportError as error:
        (output / "formal_ablation_results.xlsx.unavailable.txt").write_text(
            f"Install openpyxl and rerun this tool: {error}\n",
            encoding="utf-8",
        )
    try:
        markdown = frame.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(frame.columns) + " |"
        separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
        rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
        markdown = "\n".join([header, separator, *rows])
    (output / "formal_ablation_results.md").write_text(markdown + "\n", encoding="utf-8")
    labels = frame["Experiment"].str.split("_").str[0]
    charts = [
        (["mAP50-95"], "mAP50-95 cumulative change", "map50_95_cumulative.png"),
        (["mAP50"], "mAP50 cumulative change", "map50_cumulative.png"),
        (["P", "R"], "Precision and Recall", "precision_recall_cumulative.png"),
    ]
    for columns, title, filename in charts:
        figure, axis = plt.subplots(figsize=(8, 4))
        positions = list(range(len(frame)))
        for column in columns:
            axis.plot(positions, frame[column], marker="o", label=column)
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        if len(columns) > 1:
            axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=300)
        plt.close(figure)
    for x, filename in (("Params", "accuracy_vs_params.png"), ("GFLOPs", "accuracy_vs_gflops.png"), ("Latency", "accuracy_vs_latency.png")):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(frame[x], frame["mAP50-95"])
        for _, row in frame.iterrows():
            axis.annotate(row["Experiment"].split("_")[0], (row[x], row["mAP50-95"]))
        axis.set(xlabel=x, ylabel="mAP50-95")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=300)
        plt.close(figure)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(summarize(args.root, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
