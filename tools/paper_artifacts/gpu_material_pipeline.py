"""GPU-side paper evidence extraction from already trained checkpoints.

This module never trains a model.  It creates reusable prediction caches,
short-side-conditioned metrics, representative-case candidates, real feature
responses, CA-SCAM internal evidence, and VGUP gate statistics with fixed and
explicit inference settings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ShortSideBin:
    lower: float
    upper: float | None

    @property
    def label(self) -> str:
        if self.upper is None:
            return f"≥{self.lower:g} px"
        if self.lower == 0:
            return f"<{self.upper:g} px"
        return f"{self.lower:g}–{self.upper:g} px"

    def contains(self, value: float) -> bool:
        return value >= self.lower and (self.upper is None or value < self.upper)


DEFAULT_BINS = [
    ShortSideBin(0, 8),
    ShortSideBin(8, 16),
    ShortSideBin(16, 32),
    ShortSideBin(32, None),
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or (list(rows[0]) if rows else ["状态"])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _images_from_entry(entry: Path) -> list[Path]:
    if entry.is_dir():
        return sorted(path for path in entry.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if entry.is_file() and entry.suffix.lower() == ".txt":
        paths = []
        for line in entry.read_text(encoding="utf-8").splitlines():
            if line.strip():
                paths.append(Path(line.strip()))
        return paths
    if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
        return [entry]
    return []


def resolve_dataset_images(data_yaml: str | Path, split: str = "val") -> tuple[Path, list[Path]]:
    yaml_path = Path(data_yaml)
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    declared_root = Path(payload.get("path") or yaml_path.parent)
    if not declared_root.is_absolute():
        declared_root = (yaml_path.parent / declared_root).resolve()
    entries = payload.get(split)
    if entries is None:
        raise KeyError(f"Dataset YAML does not define split={split!r}")
    if not isinstance(entries, list):
        entries = [entries]
    images: list[Path] = []
    for entry in entries:
        path = Path(str(entry))
        if not path.is_absolute():
            path = declared_root / path
        images.extend(_images_from_entry(path))
    unique = sorted({path.resolve() for path in images})
    if not unique:
        raise FileNotFoundError(f"No images resolved for {split} from {yaml_path}")
    return declared_root, unique


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" not in lowered:
        raise ValueError(f"Image path does not contain an images directory: {image}")
    index = len(lowered) - 1 - lowered[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _ground_truth(image: Path, width: int, height: int, imgsz: int) -> list[dict[str, Any]]:
    label = _label_path(image)
    if not label.is_file():
        return []
    scale = min(imgsz / float(width), imgsz / float(height))
    records = []
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) < 5:
            raise ValueError(f"Malformed label {label}:{line_number}")
        class_id, cx, cy, box_width, box_height = values[:5]
        width_px, height_px = box_width * width, box_height * height
        x1, y1 = (cx - box_width / 2) * width, (cy - box_height / 2) * height
        x2, y2 = (cx + box_width / 2) * width, (cy + box_height / 2) * height
        records.append(
            {
                "class": int(class_id),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "short_side_at_640": float(min(width_px, height_px) * scale),
            }
        )
    return records


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    left = list(left)
    right = list(right)
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def greedy_match(
    truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            if int(prediction["class"]) != int(truth["class"]):
                continue
            candidates.append(
                (
                    box_iou(prediction["xyxy"], truth["xyxy"]),
                    float(prediction.get("confidence", 1.0)),
                    prediction_index,
                    truth_index,
                )
            )
    matched_predictions: set[int] = set()
    matched_truths: set[int] = set()
    matches = []
    for overlap, confidence, prediction_index, truth_index in sorted(candidates, reverse=True):
        if overlap < iou_threshold:
            continue
        if prediction_index in matched_predictions or truth_index in matched_truths:
            continue
        matched_predictions.add(prediction_index)
        matched_truths.add(truth_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": truth_index,
                "iou": float(overlap),
                "confidence": confidence,
            }
        )
    return matches


def _prediction_short_side(prediction: dict[str, Any], width: int, height: int, imgsz: int) -> float:
    x1, y1, x2, y2 = prediction["xyxy"]
    scale = min(imgsz / float(width), imgsz / float(height))
    return float(min(x2 - x1, y2 - y1) * scale)


def generate_prediction_cache(
    *,
    weights: str | Path,
    data_yaml: str | Path,
    output_dir: str | Path,
    model_label: str,
    split: str = "val",
    imgsz: int = 640,
    confidence_floor: float = 0.001,
    nms_iou: float = 0.7,
    device: int | str = 0,
    batch: int = 8,
) -> Path:
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    root, images = resolve_dataset_images(data_yaml, split)
    model = YOLO(str(weights))
    weights_sha256 = sha256_file(weights)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(images), desc=f"{model_label} 逐图预测", unit="张", dynamic_ncols=True)
    try:
        for start in range(0, len(images), max(1, batch)):
            chunk = images[start : start + max(1, batch)]
            results = model.predict(
                source=[str(path) for path in chunk],
                imgsz=imgsz,
                conf=confidence_floor,
                iou=nms_iou,
                device=device,
                batch=batch,
                stream=True,
                verbose=False,
            )
            for image, result in zip(chunk, results, strict=True):
                height, width = result.orig_shape
                predictions: list[dict[str, Any]] = []
                if result.boxes is not None:
                    for xyxy, confidence, class_id in zip(
                        result.boxes.xyxy.detach().cpu().tolist(),
                        result.boxes.conf.detach().cpu().tolist(),
                        result.boxes.cls.detach().cpu().tolist(),
                        strict=True,
                    ):
                        prediction = {
                            "class": int(class_id),
                            "xyxy": [float(value) for value in xyxy],
                            "confidence": float(confidence),
                        }
                        prediction["short_side_at_640"] = _prediction_short_side(
                            prediction,
                            width,
                            height,
                            imgsz,
                        )
                        predictions.append(prediction)
                truths = _ground_truth(image, width, height, imgsz)
                matches = greedy_match(truths, predictions, 0.5)
                match_by_prediction = {
                    int(item["prediction_index"]): item for item in matches
                }
                matched_truths = {
                    int(item["ground_truth_index"]) for item in matches
                }
                try:
                    relative = image.relative_to(root).as_posix()
                except ValueError:
                    relative = image.name
                for index, prediction in enumerate(predictions):
                    match = match_by_prediction.get(index)
                    flat_rows.append(
                        {
                            "image": relative,
                            "record_type": "prediction",
                            "index": index,
                            "class": prediction["class"],
                            "xyxy": json.dumps(prediction["xyxy"]),
                            "confidence": prediction["confidence"],
                            "short_side_at_640": prediction["short_side_at_640"],
                            "matched_gt": match["ground_truth_index"] if match else None,
                            "IoU": match["iou"] if match else None,
                            "TP": int(match is not None),
                            "FP": int(match is None),
                            "FN": 0,
                        }
                    )
                for index, truth in enumerate(truths):
                    flat_rows.append(
                        {
                            "image": relative,
                            "record_type": "ground_truth",
                            "index": index,
                            "class": truth["class"],
                            "xyxy": json.dumps(truth["xyxy"]),
                            "confidence": None,
                            "short_side_at_640": truth["short_side_at_640"],
                            "matched_gt": None,
                            "IoU": None,
                            "TP": 0,
                            "FP": 0,
                            "FN": int(index not in matched_truths),
                        }
                    )
                records.append(
                    {
                        "image": relative,
                        "source_path": str(image),
                        "width": int(width),
                        "height": int(height),
                        "ground_truth": truths,
                        "predictions": predictions,
                        "matches_iou50": matches,
                    }
                )
                progress.update(1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        progress.close()
    safe_label = model_label.replace("/", "_").replace(" ", "_")
    json_path = output / f"{safe_label}_{split}逐图预测缓存.json"
    _write_json(
        json_path,
        {
            "model": model_label,
            "weights": str(Path(weights)),
            "weights_sha256": weights_sha256,
            "split": split,
            "imgsz": imgsz,
            "confidence_floor": confidence_floor,
            "nms_iou": nms_iou,
            "records": records,
        },
    )
    _write_csv(
        output / f"{safe_label}_{split}逐图预测缓存.csv",
        flat_rows,
        [
            "image",
            "record_type",
            "index",
            "class",
            "xyxy",
            "confidence",
            "short_side_at_640",
            "matched_gt",
            "IoU",
            "TP",
            "FP",
            "FN",
        ],
    )
    return json_path


def load_cache(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split") not in {"val", "test"}:
        raise ValueError("Prediction cache must declare val or test split.")
    return payload


def _find_bin(value: float, bins: list[ShortSideBin]) -> ShortSideBin:
    return next(item for item in bins if item.contains(float(value)))


def grouped_counts(
    records: list[dict[str, Any]],
    *,
    bins: list[ShortSideBin] | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    groups = bins or DEFAULT_BINS
    totals = {
        item.label: {"实例数": 0, "TP": 0, "FP": 0, "FN": 0}
        for item in groups
    }
    for record in records:
        truths = record["ground_truth"]
        predictions = [
            prediction
            for prediction in record["predictions"]
            if float(prediction["confidence"]) >= confidence_threshold
        ]
        matches = greedy_match(truths, predictions, iou_threshold)
        matched_predictions = {int(item["prediction_index"]) for item in matches}
        matched_truths = {int(item["ground_truth_index"]) for item in matches}
        for index, truth in enumerate(truths):
            target = _find_bin(truth["short_side_at_640"], groups)
            totals[target.label]["实例数"] += 1
            if index in matched_truths:
                totals[target.label]["TP"] += 1
            else:
                totals[target.label]["FN"] += 1
        for index, prediction in enumerate(predictions):
            if index in matched_predictions:
                continue
            target = _find_bin(prediction["short_side_at_640"], groups)
            totals[target.label]["FP"] += 1
    rows = []
    for item in groups:
        counts = totals[item.label]
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        rows.append(
            {
                "短边分组": item.label,
                **counts,
                "Precision": tp / (tp + fp) if tp + fp else None,
                "Recall": tp / (tp + fn) if tp + fn else None,
                "计数置信度阈值": confidence_threshold,
                "匹配IoU阈值": iou_threshold,
            }
        )
    return rows


def _interpolated_ap(tp: list[int], fp: list[int], gt_count: int) -> float | None:
    if gt_count <= 0:
        return None
    if not tp:
        return 0.0
    tp_cumulative = np.cumsum(np.asarray(tp, dtype=float))
    fp_cumulative = np.cumsum(np.asarray(fp, dtype=float))
    recall = tp_cumulative / gt_count
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    grid = np.linspace(0.0, 1.0, 101)
    values = [
        float(np.max(precision[recall >= target])) if np.any(recall >= target) else 0.0
        for target in grid
    ]
    return float(np.mean(values))


def size_conditioned_ap(
    records: list[dict[str, Any]],
    target_bin: ShortSideBin,
    iou_threshold: float,
) -> float | None:
    """Compute short-side-conditioned AP with COCO-like ignore semantics.

    Ground truths outside ``target_bin`` are ignored.  A detection matched to
    such an ignored ground truth is also ignored, and an unmatched detection
    whose own short side is outside the target bin is ignored.  Remaining
    detections are globally confidence-ranked and evaluated with 101-point
    interpolated AP.  This is explicitly a custom short-side AP, not COCO's
    standard area-based AP.
    """

    gt_count = sum(
        target_bin.contains(float(truth["short_side_at_640"]))
        for record in records
        for truth in record["ground_truth"]
    )
    detections = []
    for record_index, record in enumerate(records):
        for prediction_index, prediction in enumerate(record["predictions"]):
            detections.append(
                (
                    -float(prediction["confidence"]),
                    str(record["image"]),
                    record_index,
                    prediction_index,
                )
            )
    detections.sort()
    matched_targets: dict[int, set[int]] = {index: set() for index in range(len(records))}
    tp: list[int] = []
    fp: list[int] = []
    for _negative_confidence, _image, record_index, prediction_index in detections:
        record = records[record_index]
        prediction = record["predictions"][prediction_index]
        target_candidates = []
        ignored_candidates = []
        for truth_index, truth in enumerate(record["ground_truth"]):
            if int(truth["class"]) != int(prediction["class"]):
                continue
            overlap = box_iou(prediction["xyxy"], truth["xyxy"])
            if target_bin.contains(float(truth["short_side_at_640"])):
                if truth_index not in matched_targets[record_index]:
                    target_candidates.append((overlap, truth_index))
            else:
                ignored_candidates.append((overlap, truth_index))
        best_target = max(target_candidates, default=(0.0, -1))
        if best_target[0] >= iou_threshold:
            matched_targets[record_index].add(best_target[1])
            tp.append(1)
            fp.append(0)
            continue
        best_ignored = max(ignored_candidates, default=(0.0, -1))
        if best_ignored[0] >= iou_threshold:
            continue
        if not target_bin.contains(float(prediction["short_side_at_640"])):
            continue
        tp.append(0)
        fp.append(1)
    return _interpolated_ap(tp, fp, gt_count)


def grouped_ap_rows(
    records: list[dict[str, Any]],
    bins: list[ShortSideBin] | None = None,
) -> list[dict[str, Any]]:
    groups = bins or DEFAULT_BINS
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    rows = []
    for item in groups:
        ap_values = [size_conditioned_ap(records, item, threshold) for threshold in thresholds]
        valid = [value for value in ap_values if value is not None]
        rows.append(
            {
                "短边分组": item.label,
                "AP50": ap_values[0],
                "AP50-95": float(np.mean(valid)) if valid else None,
                "GT实例数": sum(
                    item.contains(float(truth["short_side_at_640"]))
                    for record in records
                    for truth in record["ground_truth"]
                ),
                "定义": "按640 letterbox短边分组；COCO-like ignore；101点插值；非标准COCO面积AP",
            }
        )
    return rows


def _plot_group_metric(rows_by_model: dict[str, list[dict[str, Any]]], metric: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["短边分组"] for row in next(iter(rows_by_model.values()))]
    x = np.arange(len(labels))
    width = 0.8 / max(len(rows_by_model), 1)
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    colors = ["#1F4E79", "#C55A11", "#7F6000", "#7030A0"]
    for index, (model, rows) in enumerate(rows_by_model.items()):
        values = [np.nan if row.get(metric) is None else float(row[metric]) for row in rows]
        axis.bar(
            x + (index - (len(rows_by_model) - 1) / 2) * width,
            values,
            width,
            label=model,
            color=colors[index % len(colors)],
            edgecolor="#333333",
            linewidth=0.5,
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel(metric)
    axis.set_title(f"按640 letterbox短边分组的{metric}对比")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def evaluate_dpls_size_groups(
    caches: dict[str, str | Path],
    output_dir: str | Path,
    *,
    confidence_threshold: float = 0.25,
) -> Path:
    output = Path(output_dir)
    count_rows: dict[str, list[dict[str, Any]]] = {}
    ap_rows: dict[str, list[dict[str, Any]]] = {}
    combined = []
    for label, cache_path in caches.items():
        payload = load_cache(cache_path)
        records = payload["records"]
        count_rows[label] = grouped_counts(
            records,
            confidence_threshold=confidence_threshold,
        )
        ap_rows[label] = grouped_ap_rows(records)
        ap_by_group = {row["短边分组"]: row for row in ap_rows[label]}
        for row in count_rows[label]:
            combined.append({"模型": label, **row, **{key: value for key, value in ap_by_group[row["短边分组"]].items() if key != "短边分组"}})
    csv_path = _write_csv(output / "DPLS_短边分组检测结果.csv", combined)
    _plot_group_metric(count_rows, "Recall", output / "DPLS_短边分组召回率对比")
    _plot_group_metric(ap_rows, "AP50", output / "DPLS_短边分组AP50对比")
    notes = [
        "# DPLS尺寸专项结论",
        "",
        "- 分组依据：原图经过640 letterbox后的目标短边。",
        f"- TP/FP/FN与P/R使用固定置信度阈值 {confidence_threshold:g} 和 IoU≥0.5 一对一匹配。",
        "- AP50与AP50-95使用低置信度预测缓存、全局置信度排序和101点插值。",
        "- 非目标短边组GT及与其匹配的检测按COCO area-range思想忽略；这是明确标注的短边条件AP，不冒充标准COCO面积AP。",
        "- 各组实例数完整保留；不因样本稀疏而静默合并区间。",
        "- 最终论文结论必须依据本CSV真实输出后填写，Notebook不预设DPLS一定在每个尺度组领先。",
        "",
    ]
    (output / "DPLS_尺寸专项结论.md").write_text("\n".join(notes), encoding="utf-8")
    return csv_path


def _image_statistics(path: str | Path) -> tuple[float, float, float]:
    image = np.asarray(Image.open(path).convert("L").resize((128, 128)), dtype=np.float32) / 255.0
    brightness = float(image.mean())
    contrast = float(image.std())
    rms_contrast = float(np.sqrt(np.mean((image - brightness) ** 2)))
    return brightness, contrast, rms_contrast


def _record_counts(record: dict[str, Any], confidence_threshold: float) -> dict[str, Any]:
    predictions = [
        prediction for prediction in record["predictions"]
        if float(prediction["confidence"]) >= confidence_threshold
    ]
    matches = greedy_match(record["ground_truth"], predictions, 0.5)
    matched_pred = {int(item["prediction_index"]) for item in matches}
    matched_gt = {int(item["ground_truth_index"]) for item in matches}
    mean_iou = float(np.mean([item["iou"] for item in matches])) if matches else 0.0
    return {
        "tp": len(matches),
        "fp": len(predictions) - len(matched_pred),
        "fn": len(record["ground_truth"]) - len(matched_gt),
        "mean_iou": mean_iou,
        "predictions": predictions,
    }


def _draw_panel(
    image_path: str | Path,
    truths: list[dict[str, Any]],
    left_predictions: list[dict[str, Any]],
    right_predictions: list[dict[str, Any]],
    left_label: str,
    right_label: str,
    output: Path,
) -> None:
    source = Image.open(image_path).convert("RGB")

    def draw_variant(predictions: list[dict[str, Any]], title: str) -> Image.Image:
        image = source.copy()
        drawer = ImageDraw.Draw(image)
        line_width = max(2, round(max(image.size) / 320))
        for truth in truths:
            drawer.rectangle(truth["xyxy"], outline=(40, 210, 70), width=line_width)
        for prediction in predictions:
            drawer.rectangle(prediction["xyxy"], outline=(230, 55, 45), width=line_width)
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (640, 680), "white")
        canvas.paste(image, ((640 - image.width) // 2, 40))
        ImageDraw.Draw(canvas).text((10, 10), title, fill="black")
        return canvas

    panels = [draw_variant(left_predictions, left_label), draw_variant(right_predictions, right_label)]
    canvas = Image.new("RGB", (1280, 680), "white")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (640, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def compare_prediction_caches(
    *,
    left_cache: str | Path,
    right_cache: str | Path,
    output_dir: str | Path,
    prefix: str,
    left_label: str,
    right_label: str,
    confidence_threshold: float = 0.25,
    max_panels: int = 6,
) -> Path:
    left_payload, right_payload = load_cache(left_cache), load_cache(right_cache)
    left = {record["image"]: record for record in left_payload["records"]}
    right = {record["image"]: record for record in right_payload["records"]}
    rows = []
    for image in sorted(set(left).intersection(right)):
        left_record, right_record = left[image], right[image]
        left_counts = _record_counts(left_record, confidence_threshold)
        right_counts = _record_counts(right_record, confidence_threshold)
        brightness, contrast, rms = _image_statistics(left_record["source_path"])
        short_sides = [float(item["short_side_at_640"]) for item in left_record["ground_truth"]]
        categories = []
        if right_counts["fn"] < left_counts["fn"]:
            categories.append("减少漏检")
        if right_counts["fp"] < left_counts["fp"]:
            categories.append("减少误检")
        if right_counts["mean_iou"] - left_counts["mean_iou"] > 0.03:
            categories.append("定位改善")
        if short_sides and min(short_sides) < 8:
            categories.append("极小船")
        if len(short_sides) >= 3:
            categories.append("密集船")
        if contrast <= 0.12:
            categories.append("低对比度候选")
        elif contrast >= 0.25:
            categories.append("高对比度/清晰场景候选")
        if right_counts["fp"] + right_counts["fn"] > left_counts["fp"] + left_counts["fn"]:
            categories.append(f"{right_label}失败")
        improvement = (
            left_counts["fn"] + left_counts["fp"]
            - right_counts["fn"] - right_counts["fp"]
        )
        score = 3 * improvement + 2 * (right_counts["mean_iou"] - left_counts["mean_iou"])
        rows.append(
            {
                "图片名": image,
                "源路径": left_record["source_path"],
                "GT数量": len(left_record["ground_truth"]),
                f"{left_label}_TP": left_counts["tp"],
                f"{left_label}_FP": left_counts["fp"],
                f"{left_label}_FN": left_counts["fn"],
                f"{right_label}_TP": right_counts["tp"],
                f"{right_label}_FP": right_counts["fp"],
                f"{right_label}_FN": right_counts["fn"],
                "平均IoU变化": right_counts["mean_iou"] - left_counts["mean_iou"],
                "最小短边_640": min(short_sides) if short_sides else None,
                "图像平均亮度": brightness,
                "图像灰度标准差": contrast,
                "RMS对比度": rms,
                "候选类型": "；".join(categories) if categories else "无明显变化",
                "推荐得分": score,
                "需人工复核": True,
            }
        )
    rows.sort(key=lambda row: (-float(row["推荐得分"]), row["图片名"]))
    output = Path(output_dir)
    csv_path = _write_csv(output / f"{prefix}_代表案例候选.csv", rows)
    for index, row in enumerate(rows[:max_panels], start=1):
        image = row["图片名"]
        left_record, right_record = left[image], right[image]
        left_counts = _record_counts(left_record, confidence_threshold)
        right_counts = _record_counts(right_record, confidence_threshold)
        _draw_panel(
            left_record["source_path"],
            left_record["ground_truth"],
            left_counts["predictions"],
            right_counts["predictions"],
            left_label,
            right_label,
            output / f"{prefix}_检测结果对比_候选{index:03d}.png",
        )
    return csv_path


def _feature_energy(feature: torch.Tensor) -> np.ndarray:
    return feature[0].detach().float().square().mean(dim=0).cpu().numpy()


def export_dpls_feature_comparison(
    *,
    pls_weights: str | Path,
    dpls_weights: str | Path,
    image: str | Path,
    output_dir: str | Path,
    imgsz: int = 640,
    device: int | str = 0,
) -> list[dict[str, Any]]:
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    register_custom_modules()
    output = Path(output_dir)
    captured_models: dict[str, tuple[list[torch.Tensor], list[float]]] = {}
    for label, weights in (("PLS", pls_weights), ("DPLS", dpls_weights)):
        wrapper = YOLO(str(weights))
        detect = wrapper.model.model[-1]
        captured: list[torch.Tensor] = []

        def hook(_module, inputs) -> None:
            captured.clear()
            captured.extend(item.detach().cpu().clone() for item in inputs[0])

        handle = detect.register_forward_pre_hook(hook)
        try:
            wrapper.predict(source=str(image), imgsz=imgsz, device=device, verbose=False)
        finally:
            handle.remove()
        captured_models[label] = (captured, [float(value) for value in detect.stride.detach().cpu()])
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    common_strides = sorted(
        set(captured_models["PLS"][1]).intersection(captured_models["DPLS"][1])
    )
    image_stem = Path(image).stem
    for stride in common_strides:
        arrays = {}
        for label in ("PLS", "DPLS"):
            features, strides = captured_models[label]
            arrays[label] = _feature_energy(features[strides.index(stride)])
        shared_low = min(float(np.quantile(value, 0.01)) for value in arrays.values())
        shared_high = max(float(np.quantile(value, 0.99)) for value in arrays.values())
        level = int(round(math.log2(stride)))
        for label, array in arrays.items():
            path = output / f"DPLS_{label}_P{level}特征响应_图像{image_stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(path, array, cmap="magma", vmin=shared_low, vmax=max(shared_high, shared_low + 1e-12))
            rows.append(
                {
                    "模型": label,
                    "层级": f"P{level}",
                    "stride": stride,
                    "shape": list(array.shape),
                    "energy_mean": float(array.mean()),
                    "energy_std": float(array.std()),
                    "shared_vmin": shared_low,
                    "shared_vmax": shared_high,
                    "文件": path.name,
                }
            )
    _write_csv(output / "DPLS_特征响应统计.csv", rows)
    available = ", ".join(sorted({row["层级"] for row in rows}))
    (output / "DPLS_特征层级说明.md").write_text(
        "# DPLS特征层级说明\n\n"
        f"本次从真实Detect输入捕获的共同层级为：{available or '无'}。\n\n"
        "DPLS/PLS正式结构将检测层级移动至P2/P3/P4，因此模型中不存在可作为检测输入的P5；程序不会伪造P5热力图。"
        "\n\n同一层级的PLS与DPLS图使用共享1%–99%能量显示范围。\n",
        encoding="utf-8",
    )
    return rows


def _save_energy_map(tensor: torch.Tensor, path: Path, *, signed: bool = False) -> None:
    import matplotlib.pyplot as plt

    value = tensor.detach().float()
    if value.ndim == 4 and value.shape[1] != 1:
        value = value.square().mean(dim=1, keepdim=True) if not signed else value.mean(dim=1, keepdim=True)
    array = value[0, 0].cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if signed:
        bound = max(abs(float(np.quantile(array, 0.01))), abs(float(np.quantile(array, 0.99))), 1e-12)
        plt.imsave(path, array, cmap="coolwarm", vmin=-bound, vmax=bound)
    else:
        low, high = float(np.quantile(array, 0.01)), float(np.quantile(array, 0.99))
        plt.imsave(path, array, cmap="magma", vmin=low, vmax=max(high, low + 1e-12))


def export_ca_scam_debug(
    *,
    weights: str | Path,
    image: str | Path,
    output_dir: str | Path,
    imgsz: int = 640,
    device: int | str = 0,
) -> list[dict[str, Any]]:
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    wrapper = YOLO(str(weights))
    network = wrapper.model.eval()
    modules = [
        (index, layer)
        for index, layer in enumerate(network.model)
        if type(layer).__name__ in {"CASCAM", "CASCAMFixedBeta", "CASCAMUnbounded"}
    ]
    if len(modules) != 3:
        raise ValueError(f"Expected three CA-SCAM modules, found {len(modules)}")
    captured: dict[int, torch.Tensor] = {}
    handles = [
        layer.register_forward_pre_hook(
            lambda _module, inputs, index=index: captured.__setitem__(index, inputs[0].detach().clone())
        )
        for index, layer in modules
    ]
    try:
        wrapper.predict(source=str(image), imgsz=imgsz, device=device, verbose=False)
    finally:
        for handle in handles:
            handle.remove()
    output = Path(output_dir)
    rows = []
    image_stem = Path(image).stem
    for level, (index, layer) in enumerate(modules, start=2):
        feature = captured[index]
        with torch.inference_mode():
            normal = layer(feature)
            debug_output, debug = layer(feature, return_debug=True)
        difference = (normal - debug_output).abs()
        maps = {
            "输入特征能量": (feature, False),
            "SCAM上下文残差": (debug["context_residual"], True),
            "局部对比度图": (debug["local_contrast"], False),
            "空间校准图": (debug["contrast_map"], False),
            "校准后残差": (debug["calibrated_residual"], True),
            "最终输出特征": (debug_output, False),
        }
        for name, (tensor, signed) in maps.items():
            _save_energy_map(
                tensor,
                output / f"CA-SCAM_P{level}{name}_图像{image_stem}.png",
                signed=signed,
            )
        beta = layer.calibration_beta()
        rows.append(
            {
                "层级": f"P{level}",
                "模型层索引": index,
                "beta": float(beta.detach().cpu()),
                "max_delta": getattr(layer, "max_delta", None),
                "contrast_logit": float(layer.contrast_logit.detach().cpu())
                if hasattr(layer, "contrast_logit")
                else None,
                "局部对比度均值": float(debug["local_contrast"].mean()),
                "空间校准均值": float(debug["contrast_map"].mean()),
                "校准前残差绝对均值": float(debug["context_residual"].abs().mean()),
                "校准后残差绝对均值": float(debug["calibrated_residual"].abs().mean()),
                "一致性max_abs_diff": float(difference.max()),
                "一致性mean_abs_diff": float(difference.mean()),
            }
        )
    _write_csv(output / "CA-SCAM_内部机制统计.csv", rows)
    if any(float(row["一致性max_abs_diff"]) > 1e-6 for row in rows):
        raise AssertionError("CA-SCAM debug path changed the forward output.")
    return rows


def _tensor_rgb_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.uint8(array * 255))


def export_vgup_debug(
    *,
    weights: str | Path,
    image: str | Path,
    output_dir: str | Path,
    imgsz: int = 640,
    device: int | str = 0,
) -> dict[str, Any]:
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    wrapper = YOLO(str(weights))
    preprocessor = wrapper.model.model[0]
    if type(preprocessor).__name__ != "VGUPPreprocessor":
        raise ValueError("Checkpoint does not start with VGUPPreprocessor.")
    captured: list[torch.Tensor] = []
    handle = preprocessor.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        wrapper.predict(source=str(image), imgsz=imgsz, device=device, verbose=False)
    finally:
        handle.remove()
    with torch.inference_mode():
        normal = preprocessor(captured[-1])
        debug_output, debug = preprocessor(captured[-1], return_debug=True)
    difference = (normal - debug_output).abs()
    output = Path(output_dir)
    stem = Path(image).stem
    image_tensors = {
        "原始输入": captured[-1],
        "BPW原始结果": debug["bpw_image"],
        "全局门控后BPW": debug["gated_bpw_image"],
        "KBL原始结果": debug["kbl_image"],
        "最终VGUP输出": debug["output_image"],
    }
    for name, tensor in image_tensors.items():
        path = output / f"VGUP_{name}_图像{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        _tensor_rgb_image(tensor).save(path)
    _save_energy_map(
        debug["bpw_image"] - captured[-1],
        output / f"VGUP_BPW残差_图像{stem}.png",
        signed=True,
    )
    _save_energy_map(
        debug["kbl_image"] - debug["gated_bpw_image"],
        output / f"VGUP_KBL残差_图像{stem}.png",
        signed=True,
    )
    _save_energy_map(
        debug["spatial_gate"],
        output / f"VGUP_空间门热力图_图像{stem}.png",
    )
    stats = {
        "全局门控值": float(debug["global_gate"].mean()),
        "空间门mean": float(debug["spatial_gate"].mean()),
        "空间门std": float(debug["spatial_gate"].std()),
        "空间门min": float(debug["spatial_gate"].min()),
        "空间门max": float(debug["spatial_gate"].max()),
        "一致性max_abs_diff": float(difference.max()),
        "一致性mean_abs_diff": float(difference.mean()),
    }
    _write_json(output / "VGUP_单图内部机制统计.json", stats)
    if stats["一致性max_abs_diff"] > 1e-6:
        raise AssertionError("VGUP debug path changed the forward output.")
    return stats


def analyze_vgup_validation_gates(
    *,
    weights: str | Path,
    data_yaml: str | Path,
    output_dir: str | Path,
    split: str = "val",
    imgsz: int = 640,
    device: int | str = 0,
    batch: int = 8,
) -> Path:
    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    register_custom_modules()
    _root, images = resolve_dataset_images(data_yaml, split)
    wrapper = YOLO(str(weights))
    preprocessor = wrapper.model.model[0]
    if type(preprocessor).__name__ != "VGUPPreprocessor":
        raise ValueError("Checkpoint does not start with VGUPPreprocessor.")
    rows = []
    for start in tqdm(range(0, len(images), batch), desc="VGUP验证集门控统计", unit="批", dynamic_ncols=True):
        chunk = images[start : start + batch]
        captured: list[torch.Tensor] = []
        handle = preprocessor.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            list(
                wrapper.predict(
                    source=[str(path) for path in chunk],
                    imgsz=imgsz,
                    device=device,
                    batch=batch,
                    stream=True,
                    verbose=False,
                )
            )
        finally:
            handle.remove()
        # Ultralytics may execute one warm-up forward before the real batch.
        # The final capture is the real prediction batch in the official
        # predictor; retain a conservative reverse-accumulation fallback for
        # future predictor implementations that split a chunk.
        if captured and captured[-1].shape[0] == len(chunk):
            inputs = captured[-1]
        else:
            selected = []
            selected_count = 0
            for tensor in reversed(captured):
                selected.append(tensor)
                selected_count += int(tensor.shape[0])
                if selected_count >= len(chunk):
                    break
            if selected_count != len(chunk):
                raise RuntimeError(
                    f"Captured {selected_count} usable VGUP inputs for {len(chunk)} images "
                    f"from batch sizes {[int(item.shape[0]) for item in captured]}."
                )
            inputs = torch.cat(list(reversed(selected)), dim=0)
        if inputs.shape[0] != len(chunk):
            raise RuntimeError(
                f"Captured {inputs.shape[0]} VGUP inputs for {len(chunk)} images."
            )
        with torch.inference_mode():
            _outputs, debug = preprocessor(inputs, return_debug=True)
        gray = inputs.mean(dim=1)
        global_gate = debug["global_gate"].flatten(1).mean(dim=1)
        spatial = debug["spatial_gate"].flatten(1)
        for index, path in enumerate(chunk):
            values = spatial[index]
            brightness = gray[index].mean()
            centered = gray[index] - brightness
            rows.append(
                {
                    "图片名": path.name,
                    "全局门值": float(global_gate[index]),
                    "空间门mean": float(values.mean()),
                    "空间门std": float(values.std()),
                    "空间门min": float(values.min()),
                    "空间门max": float(values.max()),
                    "空间门p10": float(torch.quantile(values, 0.10)),
                    "空间门p50": float(torch.quantile(values, 0.50)),
                    "空间门p90": float(torch.quantile(values, 0.90)),
                    "图像平均亮度": float(brightness),
                    "图像灰度标准差": float(gray[index].std()),
                    "RMS contrast": float(torch.sqrt((centered.square()).mean())),
                }
            )
        del inputs, debug
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = Path(output_dir)
    csv_path = _write_csv(output / "VGUP_全验证集门控统计.csv", rows)
    arrays = {key: np.asarray([row[key] for row in rows], dtype=float) for key in rows[0] if key != "图片名"}
    plots = [
        ("全局门值", "VGUP_全局门分布", "Global gate"),
        ("空间门mean", "VGUP_空间门均值分布", "Spatial gate mean"),
        ("空间门std", "VGUP_空间门离散程度分布", "Spatial gate std"),
    ]
    for key, filename, xlabel in plots:
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        axis.hist(arrays[key], bins=30, color="#1F4E79", edgecolor="white")
        axis.set(xlabel=xlabel, ylabel="Images", title=filename)
        axis.spines[["top", "right"]].set_visible(False)
        figure.savefig(output / f"{filename}.png", dpi=300, bbox_inches="tight")
        figure.savefig(output / f"{filename}.pdf", bbox_inches="tight")
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    axis.scatter(arrays["RMS contrast"], arrays["空间门mean"], s=12, alpha=0.55, color="#C55A11")
    axis.set(xlabel="RMS contrast", ylabel="Spatial gate mean", title="VGUP门控与图像对比度关系")
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "VGUP_门控与图像对比度关系.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "VGUP_门控与图像对比度关系.pdf", bbox_inches="tight")
    plt.close(figure)
    correlations = {
        "global_vs_brightness_pearson": float(np.corrcoef(arrays["全局门值"], arrays["图像平均亮度"])[0, 1]),
        "global_vs_rms_contrast_pearson": float(np.corrcoef(arrays["全局门值"], arrays["RMS contrast"])[0, 1]),
        "spatial_mean_vs_rms_contrast_pearson": float(np.corrcoef(arrays["空间门mean"], arrays["RMS contrast"])[0, 1]),
    }
    (output / "VGUP_门控统计分析.md").write_text(
        "# VGUP门控统计分析\n\n"
        f"- 验证集图像数：{len(rows)}。\n"
        f"- 全局门与平均亮度Pearson相关：{correlations['global_vs_brightness_pearson']:.6f}。\n"
        f"- 全局门与RMS对比度Pearson相关：{correlations['global_vs_rms_contrast_pearson']:.6f}。\n"
        f"- 空间门均值与RMS对比度Pearson相关：{correlations['spatial_mean_vs_rms_contrast_pearson']:.6f}。\n\n"
        "上述为描述性相关，不预设单调关系，也不把低对比度候选解释为真实天气标签。\n",
        encoding="utf-8",
    )
    return csv_path


def package_gpu_results(output_root: str | Path, zip_path: str | Path) -> Path:
    root = Path(output_root)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_base = destination.with_suffix("")
    generated = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=str(root.parent),
            base_dir=root.name,
        )
    )
    if generated != destination:
        generated.replace(destination)
    return destination


__all__ = [
    "DEFAULT_BINS",
    "ShortSideBin",
    "analyze_vgup_validation_gates",
    "box_iou",
    "compare_prediction_caches",
    "evaluate_dpls_size_groups",
    "export_ca_scam_debug",
    "export_dpls_feature_comparison",
    "export_vgup_debug",
    "generate_prediction_cache",
    "greedy_match",
    "grouped_ap_rows",
    "grouped_counts",
    "load_cache",
    "package_gpu_results",
    "resolve_dataset_images",
    "sha256_file",
    "size_conditioned_ap",
]
