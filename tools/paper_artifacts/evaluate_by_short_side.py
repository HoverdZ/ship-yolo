"""Evaluate saved real predictions by 640-letterbox short-side groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_artifacts.grouped_evaluation import (
    Bin,
    evaluate_groups,
    load_records,
    recommend_merged_bins,
)
from tools.paper_artifacts.results.common import write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[8, 16, 32])
    parser.add_argument("--minimum-instances", type=int, default=30)
    args = parser.parse_args()
    records = load_records(args.predictions_json)
    values = [
        float(truth["short_side_at_640"])
        for record in records
        for truth in record["ground_truth"]
    ]
    recommendation = recommend_merged_bins(
        values,
        args.thresholds,
        minimum_instances=args.minimum_instances,
    )
    bins = [
        Bin(float(item["lower"]), item["upper"])
        for item in recommendation["recommended_bins"]
    ]

    def prediction_short_side(record, prediction) -> float:
        x1, y1, x2, y2 = prediction["xyxy"]
        scale = 640.0 / max(float(record["width"]), float(record["height"]))
        return min(x2 - x1, y2 - y1) * scale

    rows = evaluate_groups(
        records,
        bins,
        lambda truth: float(truth["short_side_at_640"]),
        prediction_short_side,
    )
    prefix = Path(args.output)
    write_rows(rows, prefix, title="Short-side-stratified detection")
    recommendation_path = prefix.with_name(prefix.name + "_bin_recommendation.json")
    recommendation_path.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(recommendation_path)


if __name__ == "__main__":
    main()
