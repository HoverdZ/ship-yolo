"""Evaluate saved real predictions by box area after 640 letterbox scaling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_artifacts.grouped_evaluation import (
    evaluate_groups,
    load_records,
    make_bins,
)
from tools.paper_artifacts.results.common import write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[64, 256, 1024],
    )
    args = parser.parse_args()
    records = load_records(args.predictions_json)
    bins = make_bins(args.thresholds)

    def gt_area(truth) -> float:
        x1, y1, x2, y2 = truth["xyxy"]
        # Ground-truth coordinates are in the original image. The record-level
        # scale is needed, so area is injected below through a closure wrapper.
        return (x2 - x1) * (y2 - y1)

    expanded = []
    for record in records:
        scale = 640.0 / max(float(record["width"]), float(record["height"]))
        clone = dict(record)
        clone["ground_truth"] = [
            {**truth, "area_at_640": gt_area(truth) * scale * scale}
            for truth in record["ground_truth"]
        ]
        expanded.append(clone)

    def prediction_area(record, prediction) -> float:
        x1, y1, x2, y2 = prediction["xyxy"]
        scale = 640.0 / max(float(record["width"]), float(record["height"]))
        return (x2 - x1) * (y2 - y1) * scale * scale

    rows = evaluate_groups(
        expanded,
        bins,
        lambda truth: float(truth["area_at_640"]),
        prediction_area,
    )
    write_rows(rows, Path(args.output), title="Area-stratified detection")


if __name__ == "__main__":
    main()
