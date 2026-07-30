"""Deterministic grouped TP/FP/FN evaluation from saved real predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float | None

    @property
    def label(self) -> str:
        if self.upper is None:
            return f">={self.lower:g}"
        return f"[{self.lower:g},{self.upper:g})"

    def contains(self, value: float) -> bool:
        return value >= self.lower and (
            self.upper is None or value < self.upper
        )


def make_bins(thresholds: list[float]) -> list[Bin]:
    edges = [0.0, *sorted(set(float(value) for value in thresholds))]
    return [
        Bin(edges[index], edges[index + 1])
        for index in range(len(edges) - 1)
    ] + [Bin(edges[-1], None)]


def recommend_merged_bins(
    values: list[float],
    thresholds: list[float],
    *,
    minimum_instances: int,
) -> dict[str, Any]:
    bins = make_bins(thresholds)
    counts = [sum(item.contains(value) for value in values) for item in bins]
    merged: list[Bin] = []
    running_lower = bins[0].lower
    running_count = 0
    notes = []
    for index, (item, count) in enumerate(zip(bins, counts, strict=True)):
        running_count += count
        is_last = index == len(bins) - 1
        if running_count >= minimum_instances or is_last:
            merged.append(Bin(running_lower, item.upper))
            if running_lower != item.lower:
                notes.append(
                    f"Merged sparse adjacent bins into {merged[-1].label} "
                    f"({running_count} instances)."
                )
            running_lower = item.upper if item.upper is not None else item.lower
            running_count = 0
    if len(merged) > 1:
        last_count = sum(
            item.contains(value) for value in values for item in merged[-1:]
        )
        if last_count < minimum_instances:
            previous = merged[-2]
            last = merged[-1]
            merged[-2:] = [Bin(previous.lower, last.upper)]
            notes.append(
                f"Merged final sparse bin into {merged[-1].label} "
                f"({last_count} instances before merge)."
            )
    return {
        "candidate_bins": [item.label for item in bins],
        "candidate_counts": counts,
        "recommended_bins": [
            {"lower": item.lower, "upper": item.upper, "label": item.label}
            for item in merged
        ],
        "minimum_instances": minimum_instances,
        "merge_notes": notes,
    }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split") not in {"val", "test"}:
        raise ValueError("Prediction file must declare val or test split.")
    return payload["records"]


def evaluate_groups(
    records: list[dict[str, Any]],
    bins: list[Bin],
    gt_measure: Callable[[dict[str, Any]], float],
    prediction_measure: Callable[[dict[str, Any], dict[str, Any]], float],
) -> list[dict[str, Any]]:
    totals = {
        item.label: {"instances": 0, "tp": 0, "fp": 0, "fn": 0}
        for item in bins
    }
    for record in records:
        gt_bins: dict[int, str] = {}
        for index, truth in enumerate(record["ground_truth"]):
            value = gt_measure(truth)
            target = next(item for item in bins if item.contains(value))
            gt_bins[index] = target.label
            totals[target.label]["instances"] += 1
        matched_predictions = set()
        for match in record.get("matches", []):
            prediction_index = int(match["prediction_index"])
            truth_index = int(match["ground_truth_index"])
            matched_predictions.add(prediction_index)
            totals[gt_bins[truth_index]]["tp"] += 1
        matched_gt = {
            int(match["ground_truth_index"])
            for match in record.get("matches", [])
        }
        for index in range(len(record["ground_truth"])):
            if index not in matched_gt:
                totals[gt_bins[index]]["fn"] += 1
        for index, prediction in enumerate(record["predictions"]):
            if index in matched_predictions:
                continue
            value = prediction_measure(record, prediction)
            target = next(item for item in bins if item.contains(value))
            totals[target.label]["fp"] += 1
    rows = []
    for item in bins:
        counts = totals[item.label]
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        rows.append(
            {
                "group": item.label,
                "lower": item.lower,
                "upper": item.upper,
                **counts,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "evaluation": "IoU>=0.5 greedy one-to-one matching",
                "ap": None,
                "ap_note": (
                    "Grouped AP is intentionally not inferred from grouped "
                    "precision/recall; use a separately validated AP implementation."
                ),
            }
        )
    return rows


__all__ = [
    "Bin",
    "evaluate_groups",
    "load_records",
    "make_bins",
    "recommend_merged_bins",
]
