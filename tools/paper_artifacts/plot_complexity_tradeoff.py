"""Plot all eligible configured models without favorable-result filtering."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _plot(rows, x, y, output: Path, title: str, ideal: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for row in rows:
        axis.scatter(row[x], row[y], s=45)
        axis.annotate(row["name"], (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(xlabel=x, ylabel=y, title=title)
    axis.grid(alpha=0.25)
    axis.text(
        0.99,
        0.01,
        f"Ideal direction: {ideal}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    models = payload.get("models", [])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"included": {}, "omitted_missing_values": {}, "speed_groups": {}}
    for x, ideal in (("gflops", "upper-left"), ("params", "upper-left")):
        rows = [row for row in models if _number(row.get(x)) and _number(row.get("map5095"))]
        report["included"][x] = [row["name"] for row in rows]
        report["omitted_missing_values"][x] = [row.get("name") for row in models if row not in rows]
        if rows:
            _plot(rows, x, "map5095", output / f"accuracy_{x}.png", f"Accuracy vs {x}", ideal)
    speed_groups = defaultdict(list)
    for row in models:
        if not (_number(row.get("fps")) and _number(row.get("map5095"))):
            continue
        key = (
            row.get("hardware"),
            row.get("inference_framework"),
            row.get("precision"),
            row.get("input_size"),
            row.get("includes_preprocess"),
            row.get("includes_nms"),
        )
        speed_groups[key].append(row)
    for key, rows in speed_groups.items():
        name = _safe("_".join(str(value) for value in key))
        report["speed_groups"][name] = {
            "protocol": key,
            "models": [row["name"] for row in rows],
        }
        _plot(
            rows,
            "fps",
            "map5095",
            output / f"accuracy_fps_{name}.png",
            "Accuracy vs FPS",
            "upper-right",
        )
    (output / "complexity_plot_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
