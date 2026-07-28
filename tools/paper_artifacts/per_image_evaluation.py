"""Save reproducible validation predictions and image-level error metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from tools.paper_artifacts.formal_protocol import IMAGE_SUFFIXES, FormalConfig, write_json


def _images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    if path.is_file() and path.suffix.lower() == ".txt":
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [path]


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _ground_truth(image: Path, width: int, height: int) -> list[dict[str, Any]]:
    label = _label_path(image)
    records = []
    if not label.is_file():
        return records
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(item) for item in line.split()]
        cls, cx, cy, box_width, box_height = values[:5]
        x1, y1 = (cx - box_width / 2) * width, (cy - box_height / 2) * height
        x2, y2 = (cx + box_width / 2) * width, (cy + box_height / 2) * height
        records.append({"class": int(cls), "xyxy": [x1, y1, x2, y2], "short_side_at_640": min(box_width, box_height) * 640})
    return records


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-12)


def _match(gt: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float = 0.5) -> tuple[int, int, int, list[int], list[int]]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, truth in enumerate(gt):
            if prediction["class"] == truth["class"]:
                candidates.append((_iou(prediction["xyxy"], truth["xyxy"]), pred_index, gt_index))
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for overlap, pred_index, gt_index in sorted(candidates, reverse=True):
        if overlap < threshold or pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)
    tp = len(matched_pred)
    return tp, len(predictions) - tp, len(gt) - tp, sorted(matched_pred), sorted(matched_gt)


def _size_bucket(short_side: float, config: FormalConfig) -> str:
    if short_side < config.tiny_short_side:
        return "tiny"
    if short_side < config.small_short_side:
        return "small"
    return "medium_large"


def evaluate_per_image(config: FormalConfig, model) -> dict[str, Any]:
    payload = yaml.safe_load(config.local_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    entries = payload["val"] if isinstance(payload["val"], list) else [payload["val"]]
    images = sorted({item.resolve() for entry in entries for item in _images(Path(entry) if Path(entry).is_absolute() else root / entry)})
    results = model.predict(
        source=[str(image) for image in images],
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        device=config.device,
        stream=True,
        verbose=False,
    )
    records: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    size_totals = {name: {"gt": 0, "tp": 0, "fn": 0} for name in ("tiny", "small", "medium_large")}
    for image, result in zip(images, results, strict=True):
        height, width = result.orig_shape
        boxes = result.boxes
        predictions = []
        if boxes is not None:
            for xyxy, confidence, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), strict=True):
                predictions.append({"xyxy": [float(value) for value in xyxy], "confidence": float(confidence), "class": int(cls)})
        gt = _ground_truth(image, width, height)
        tp, fp, fn, matched_pred, matched_gt = _match(gt, predictions)
        for gt_index, truth in enumerate(gt):
            bucket = _size_bucket(truth["short_side_at_640"], config)
            size_totals[bucket]["gt"] += 1
            if gt_index in matched_gt:
                size_totals[bucket]["tp"] += 1
            else:
                size_totals[bucket]["fn"] += 1
        relative = image.relative_to(root).as_posix()
        record = {
            "image": relative,
            "source_path": str(image),
            "width": width,
            "height": height,
            "ground_truth": gt,
            "predictions": predictions,
            "gt_count": len(gt),
            "prediction_count": len(predictions),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else (1.0 if not gt else 0.0),
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "has_miss": fn > 0,
            "has_false_positive": fp > 0,
            "matched_prediction_indices": matched_pred,
        }
        records.append(record)
        if predictions:
            for index, prediction in enumerate(predictions):
                prediction_rows.append({"image": relative, "prediction_index": index, **prediction})
        else:
            prediction_rows.append({"image": relative, "prediction_index": "", "xyxy": "", "confidence": "", "class": ""})
    output = {
        "experiment_id": config.experiment_id,
        "split": "val",
        "selection_data_only": True,
        "confidence_threshold": config.conf,
        "nms_iou_threshold": config.iou,
        "matching_iou_threshold": 0.5,
        "images": len(records),
        "records": records,
    }
    write_json(config.run_dir / "val_predictions.json", output)
    with (config.run_dir / "val_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "prediction_index", "xyxy", "confidence", "class"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    metric_fields = ["image", "gt_count", "prediction_count", "tp", "fp", "fn", "precision", "recall", "has_miss", "has_false_positive"]
    with (config.run_dir / "val_image_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows({key: record[key] for key in metric_fields} for record in records)
    with (config.run_dir / "size_stratified_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["size", "short_side_min", "short_side_max", "gt", "tp", "fn", "recall"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        bounds = {
            "tiny": (0, config.tiny_short_side),
            "small": (config.tiny_short_side, config.small_short_side),
            "medium_large": (config.small_short_side, ""),
        }
        for name, counts in size_totals.items():
            lower, upper = bounds[name]
            writer.writerow({"size": name, "short_side_min": lower, "short_side_max": upper, **counts, "recall": counts["tp"] / counts["gt"] if counts["gt"] else ""})
    return output


__all__ = ["evaluate_per_image"]
