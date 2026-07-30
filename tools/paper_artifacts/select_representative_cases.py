"""Rank representative and failure cases from objective per-image deltas."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {record["image"]: record for record in payload["records"]}


def _mean_iou(record: dict[str, Any]) -> float:
    values = [float(item["iou"]) for item in record.get("matches", [])]
    return sum(values) / len(values) if values else 0.0


def _image_stats(path: str) -> tuple[float, float]:
    array = np.asarray(
        Image.open(path).convert("L").resize((128, 128)),
        dtype=np.float32,
    ) / 255.0
    return float(array.mean()), float(array.std())


def _rank(
    rows: list[dict[str, Any]],
    category: str,
    key: str,
    *,
    ascending: bool,
    count: int,
    condition=lambda _row: True,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if condition(row)]
    candidates.sort(key=lambda row: (row[key], row["image"]), reverse=not ascending)
    return [
        {
            **row,
            "selection_category": category,
            "category_rank": rank,
            "selection_score": row[key],
            "manual_confirmation_required": True,
        }
        for rank, row in enumerate(candidates[:count], start=1)
    ]


def _draw(
    record: dict[str, Any],
    *,
    show_gt: bool,
    title: str,
    size: tuple[int, int] = (480, 320),
) -> Image.Image:
    image = Image.open(record["source_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    if show_gt:
        for truth in record["ground_truth"]:
            draw.rectangle(truth["xyxy"], outline=(40, 220, 70), width=3)
    for prediction in record["predictions"]:
        draw.rectangle(prediction["xyxy"], outline=(235, 50, 45), width=3)
    image.thumbnail(size)
    canvas = Image.new("RGB", (size[0], size[1] + 34), "white")
    canvas.paste(image, ((size[0] - image.width) // 2, 34))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black")
    return canvas


def _contact_sheet(
    selected: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    pages = []
    for row in selected:
        image = row["image"]
        panels = [
            _draw(
                {**baseline[image], "predictions": []},
                show_gt=True,
                title=f"{row['selection_category']} | GT",
            ),
            _draw(baseline[image], show_gt=False, title="Baseline"),
            _draw(final[image], show_gt=False, title="Final"),
        ]
        page = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
            "white",
        )
        x = 0
        for panel in panels:
            page.paste(panel, (x, 0))
            x += panel.width
        pages.append(page)
    if not pages:
        pages = [Image.new("RGB", (800, 300), "white")]
    output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output, save_all=True, append_images=pages[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()
    baseline = _load(args.baseline)
    final = _load(args.final)
    common = sorted(set(baseline).intersection(final))
    rows = []
    for image in common:
        base, improved = baseline[image], final[image]
        brightness, contrast = _image_stats(base["source_path"])
        short_sides = [
            float(truth["short_side_at_640"])
            for truth in base["ground_truth"]
        ]
        rows.append(
            {
                "image": image,
                "source_path": base["source_path"],
                "gt_count": base["gt_count"],
                "minimum_short_side_640": min(short_sides) if short_sides else None,
                "brightness": brightness,
                "contrast": contrast,
                "baseline_fn": base["fn"],
                "final_fn": improved["fn"],
                "baseline_fp": base["fp"],
                "final_fp": improved["fp"],
                "fn_delta": improved["fn"] - base["fn"],
                "fp_delta": improved["fp"] - base["fp"],
                "iou_delta": _mean_iou(improved) - _mean_iou(base),
                "final_total_error": improved["fn"] + improved["fp"],
            }
        )
    n = args.per_category
    selected = []
    selected += _rank(rows, "baseline_miss_final_detect", "fn_delta", ascending=True, count=n, condition=lambda row: row["fn_delta"] < 0)
    selected += _rank(rows, "baseline_fp_final_removed", "fp_delta", ascending=True, count=n, condition=lambda row: row["fp_delta"] < 0)
    selected += _rank(rows, "localization_iou_improved", "iou_delta", ascending=False, count=n, condition=lambda row: row["iou_delta"] > 0)
    selected += _rank(rows, "tiny_ship", "minimum_short_side_640", ascending=True, count=n, condition=lambda row: row["minimum_short_side_640"] is not None)
    selected += _rank(rows, "dense_ship", "gt_count", ascending=False, count=n)
    selected += _rank(rows, "low_contrast_candidate", "contrast", ascending=True, count=n)
    selected += _rank(rows, "low_illumination_candidate", "brightness", ascending=True, count=n)
    selected += _rank(rows, "final_failure", "final_total_error", ascending=False, count=n, condition=lambda row: row["final_total_error"] > 0)
    # Keep one row per category/image pair; the same image may legitimately
    # appear under more than one objective reason.
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "representative_case_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]) if selected else ["image"])
        writer.writeheader()
        writer.writerows(selected)
    _contact_sheet(
        selected,
        baseline,
        final,
        output / "representative_case_contact_sheet.pdf",
    )
    (output / "selection_policy.json").write_text(
        json.dumps(
            {
                "split": "from input prediction manifests",
                "same_images_thresholds_and_display_range": True,
                "automatic_output_is_candidate_list_only": True,
                "manual_final_selection_required": True,
                "includes_failure_cases": True,
                "scene_labels_are_objective_image-statistic candidates, not semantic ground truth": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(csv_path)


if __name__ == "__main__":
    main()
