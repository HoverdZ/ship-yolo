"""Read-only YOLO dataset parsing and scale geometry."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
INSTANCE_COLUMNS = [
    "split",
    "image_relative_path",
    "label_relative_path",
    "image_width",
    "image_height",
    "class_id",
    "box_index",
    "x_center_norm",
    "y_center_norm",
    "width_norm",
    "height_norm",
    "letterbox_scale",
    "width_original_px",
    "height_original_px",
    "width_640_px",
    "height_640_px",
    "short_side_640_px",
    "long_side_640_px",
    "area_640_px2",
    "aspect_ratio",
]


def quantile_linear(values: Sequence[float] | np.ndarray, probabilities: Sequence[float]) -> np.ndarray:
    """Return deterministic NumPy linear-interpolation quantiles."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot compute quantiles from empty data.")
    if not np.isfinite(array).all():
        raise ValueError("Quantile input contains non-finite values.")
    return np.quantile(array, np.asarray(probabilities, dtype=np.float64), method="linear")


def box_geometry(
    image_width: int,
    image_height: int,
    width_norm: float,
    height_norm: float,
    imgsz: int = 640,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Recover a YOLO box and map its size through aspect-preserving letterbox scaling."""
    if image_width <= 0 or image_height <= 0 or imgsz <= 0:
        raise ValueError("Image dimensions and imgsz must be positive.")
    if width_norm <= 0 or height_norm <= 0:
        raise ValueError("Normalized box width and height must be positive.")
    scale = min(imgsz / float(image_width), imgsz / float(image_height))
    width_original = float(width_norm) * float(image_width)
    height_original = float(height_norm) * float(image_height)
    width_input = scale * width_original
    height_input = scale * height_original
    short_side = min(width_input, height_input)
    long_side = max(width_input, height_input)
    return {
        "letterbox_scale": scale,
        "width_original_px": width_original,
        "height_original_px": height_original,
        "width_640_px": width_input,
        "height_640_px": height_input,
        "short_side_640_px": short_side,
        "long_side_640_px": long_side,
        "area_640_px2": width_input * height_input,
        "aspect_ratio": long_side / max(short_side, epsilon),
    }


def dilution_metrics(short_side_quantile_px: float, stride: int) -> dict[str, float | bool]:
    """Compute intervals spanned and the bounded spatial-dilution rate."""
    if short_side_quantile_px < 0 or stride <= 0:
        raise ValueError("Short-side length must be non-negative and stride must be positive.")
    ratio = float(short_side_quantile_px) / float(stride)
    rate = max(0.0, min(100.0, (1.0 - ratio) * 100.0))
    return {
        "sampling_intervals_spanned": ratio,
        "dilution_rate_percent": rate,
        "representable_by_one_interval": bool(short_side_quantile_px >= stride),
    }


def _load_data_yaml(data_yaml: Path) -> dict[str, Any]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Data YAML must contain a mapping: {data_yaml}")
    missing = [split for split in SPLITS if split not in payload]
    if missing:
        raise ValueError(f"Data YAML is missing required splits: {missing}")
    return payload


def _yaml_root(data_yaml: Path, payload: dict[str, Any], dataset_root: Path | None) -> Path:
    if dataset_root is not None:
        root = dataset_root.expanduser().resolve()
    else:
        configured = Path(str(payload.get("path", data_yaml.parent)))
        root = configured if configured.is_absolute() else data_yaml.parent / configured
        root = root.expanduser().resolve()
        if not root.exists() and configured.is_absolute():
            fallback = data_yaml.parent.resolve()
            if all((fallback / str(payload[split])).exists() for split in SPLITS):
                root = fallback
    if not root.is_dir():
        raise FileNotFoundError(f"Resolved dataset root is not a directory: {root}")
    return root


def _resolve_config_path(raw: str | Path, root: Path, data_yaml: Path, configured_root: str | None) -> Path:
    path = Path(str(raw)).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        if configured_root:
            try:
                relative = path.relative_to(Path(configured_root))
                candidates.append(root / relative)
            except (ValueError, OSError):
                pass
        candidates.append(root / path.name)
    else:
        candidates.extend((root / path, data_yaml.parent / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_list_entry(raw: str, list_file: Path, root: Path) -> Path:
    entry = Path(raw.strip().strip("\"'")).expanduser()
    candidates = [entry] if entry.is_absolute() else [list_file.parent / entry, root / entry]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_split_images(
    data_yaml: str | Path,
    split: str,
    dataset_root: str | Path | None = None,
) -> tuple[Path, list[Path], list[dict[str, str]]]:
    """Resolve a split defined by directories, text lists, relative paths, or absolute paths."""
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    yaml_path = Path(data_yaml).expanduser().resolve()
    payload = _load_data_yaml(yaml_path)
    root = _yaml_root(yaml_path, payload, Path(dataset_root) if dataset_root is not None else None)
    entries = payload[split] if isinstance(payload[split], list) else [payload[split]]
    images: list[Path] = []
    issues: list[dict[str, str]] = []
    for raw in entries:
        source = _resolve_config_path(str(raw), root, yaml_path, str(payload.get("path", "")) or None)
        if source.is_dir():
            images.extend(path.resolve() for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                image = _resolve_list_entry(stripped, source, root)
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    issues.append(
                        {
                            "split": split,
                            "issue_type": "unsupported_image_suffix",
                            "relative_path": _stable_relative(image, root),
                            "detail": f"{source.name}:{line_number}",
                        }
                    )
                else:
                    images.append(image)
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source.resolve())
        else:
            issues.append(
                {
                    "split": split,
                    "issue_type": "unresolved_split_source",
                    "relative_path": _stable_relative(source, root),
                    "detail": str(raw),
                }
            )
    unique: dict[str, Path] = {}
    for image in sorted(images, key=lambda item: item.as_posix().casefold()):
        key = str(image).casefold()
        if key in unique:
            issues.append(
                {
                    "split": split,
                    "issue_type": "duplicate_image_reference",
                    "relative_path": _stable_relative(image, root),
                    "detail": "duplicate split entry ignored",
                }
            )
            continue
        unique[key] = image
    return root, list(unique.values()), issues


def _stable_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.name


def _label_path(image: Path, root: Path, split: str) -> Path:
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].casefold() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return root / split / "labels" / f"{image.stem}.txt"


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    if width <= 0 or height <= 0:
        raise ValueError("non-positive image dimensions")
    return int(width), int(height)


def _parse_label_line(line: str, line_number: int) -> tuple[tuple[int, float, float, float, float] | None, str | None]:
    tokens = line.split()
    if len(tokens) != 5:
        return None, f"line {line_number}: expected 5 columns, got {len(tokens)}"
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return None, f"line {line_number}: non-numeric value"
    if not all(math.isfinite(value) for value in values):
        return None, f"line {line_number}: non-finite value"
    class_value, x_center, y_center, width, height = values
    if class_value < 0 or not class_value.is_integer():
        return None, f"line {line_number}: invalid class id {class_value}"
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None, f"line {line_number}: center outside [0,1]"
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return None, f"line {line_number}: width/height outside (0,1]"
    tolerance = 1e-9
    if (
        x_center - width / 2 < -tolerance
        or x_center + width / 2 > 1 + tolerance
        or y_center - height / 2 < -tolerance
        or y_center + height / 2 > 1 + tolerance
    ):
        return None, f"line {line_number}: box extends outside normalized image bounds"
    return (int(class_value), x_center, y_center, width, height), None


def _metric_summary(frame: pd.DataFrame, metric: str) -> dict[str, float | None]:
    if frame.empty:
        return {"min": None, "max": None, "mean": None, "median": None}
    values = frame[metric].to_numpy(dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _audit_row(split: str, image_frame: pd.DataFrame, instance_frame: pd.DataFrame, issue_frame: pd.DataFrame) -> dict[str, Any]:
    readable = image_frame[image_frame["readable"]]
    invalid_label_rows = issue_frame[issue_frame["issue_type"] == "abnormal_label_row"]
    row: dict[str, Any] = {
        "split": split,
        "image_count": int(len(image_frame)),
        "label_file_present_count": int(image_frame["label_exists"].sum()) if not image_frame.empty else 0,
        "labeled_image_count": int((image_frame["valid_instances"] > 0).sum()) if not image_frame.empty else 0,
        "empty_label_image_count": int(image_frame["label_empty"].sum()) if not image_frame.empty else 0,
        "instance_count": int(len(instance_frame)),
        "unreadable_image_count": int((~image_frame["readable"]).sum()) if not image_frame.empty else 0,
        "abnormal_label_count": int(len(invalid_label_rows)),
        "image_width_min": int(readable["image_width"].min()) if not readable.empty else None,
        "image_width_max": int(readable["image_width"].max()) if not readable.empty else None,
        "image_height_min": int(readable["image_height"].min()) if not readable.empty else None,
        "image_height_max": int(readable["image_height"].max()) if not readable.empty else None,
    }
    for metric, prefix in (
        ("short_side_640_px", "short_side_640_px"),
        ("long_side_640_px", "long_side_640_px"),
        ("area_640_px2", "area_640_px2"),
        ("aspect_ratio", "aspect_ratio"),
    ):
        for statistic, value in _metric_summary(instance_frame, metric).items():
            row[f"{prefix}_{statistic}"] = value
    return row


def analyze_dataset(
    data_yaml: str | Path,
    imgsz: int = 640,
    dataset_root: str | Path | None = None,
    source_label: str = "configured data YAML",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Analyze all frozen splits without modifying any image, label, or YAML file."""
    yaml_path = Path(data_yaml).expanduser().resolve()
    all_instances: list[dict[str, Any]] = []
    all_images: list[dict[str, Any]] = []
    all_issues: list[dict[str, str]] = []
    roots: list[Path] = []
    fingerprint = hashlib.sha256()

    for split in SPLITS:
        root, images, resolution_issues = resolve_split_images(yaml_path, split, dataset_root)
        roots.append(root)
        all_issues.extend(resolution_issues)
        for image in images:
            relative_image = _stable_relative(image, root)
            label = _label_path(image, root, split)
            relative_label = _stable_relative(label, root)
            fingerprint.update(f"{split}\0{relative_image}\0".encode("utf-8"))
            try:
                fingerprint.update(str(image.stat().st_size).encode("ascii"))
            except OSError:
                fingerprint.update(b"missing")
            try:
                width, height = _read_image_size(image)
                readable = True
            except Exception as error:
                width = height = 0
                readable = False
                all_issues.append(
                    {
                        "split": split,
                        "issue_type": "unreadable_image",
                        "relative_path": relative_image,
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )
            label_exists = label.is_file()
            label_empty = bool(label_exists and label.stat().st_size == 0)
            valid_instances = 0
            invalid_rows = 0
            if not label_exists:
                all_issues.append(
                    {
                        "split": split,
                        "issue_type": "missing_label_file",
                        "relative_path": relative_label,
                        "detail": f"image={relative_image}",
                    }
                )
            elif readable:
                try:
                    lines = label.read_text(encoding="utf-8-sig").splitlines()
                except Exception as error:
                    lines = []
                    all_issues.append(
                        {
                            "split": split,
                            "issue_type": "unreadable_label",
                            "relative_path": relative_label,
                            "detail": f"{type(error).__name__}: {error}",
                        }
                    )
                for line_number, raw_line in enumerate(lines, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    parsed, error = _parse_label_line(stripped, line_number)
                    if error:
                        invalid_rows += 1
                        all_issues.append(
                            {
                                "split": split,
                                "issue_type": "abnormal_label_row",
                                "relative_path": relative_label,
                                "detail": error,
                            }
                        )
                        continue
                    class_id, x_center, y_center, width_norm, height_norm = parsed
                    geometry = box_geometry(width, height, width_norm, height_norm, imgsz=imgsz)
                    row = {
                        "split": split,
                        "image_relative_path": relative_image,
                        "label_relative_path": relative_label,
                        "image_width": width,
                        "image_height": height,
                        "class_id": class_id,
                        "box_index": line_number - 1,
                        "x_center_norm": x_center,
                        "y_center_norm": y_center,
                        "width_norm": width_norm,
                        "height_norm": height_norm,
                        **geometry,
                    }
                    if imgsz != 640:
                        row["width_640_px"] = geometry["width_640_px"]
                        row["height_640_px"] = geometry["height_640_px"]
                        row["short_side_640_px"] = geometry["short_side_640_px"]
                        row["long_side_640_px"] = geometry["long_side_640_px"]
                        row["area_640_px2"] = geometry["area_640_px2"]
                    all_instances.append(row)
                    valid_instances += 1
            all_images.append(
                {
                    "split": split,
                    "image_relative_path": relative_image,
                    "readable": readable,
                    "image_width": width if readable else None,
                    "image_height": height if readable else None,
                    "label_exists": label_exists,
                    "label_empty": label_empty,
                    "valid_instances": valid_instances,
                    "invalid_rows": invalid_rows,
                }
            )

    instances = pd.DataFrame(all_instances, columns=INSTANCE_COLUMNS)
    image_records = pd.DataFrame(all_images)
    issues = pd.DataFrame(all_issues, columns=["split", "issue_type", "relative_path", "detail"])
    audit_rows: list[dict[str, Any]] = []
    for split in (*SPLITS, "all"):
        image_subset = image_records if split == "all" else image_records[image_records["split"] == split]
        instance_subset = instances if split == "all" else instances[instances["split"] == split]
        issue_subset = issues if split == "all" else issues[issues["split"] == split]
        audit_rows.append(_audit_row(split, image_subset, instance_subset, issue_subset))
    audit = pd.DataFrame(audit_rows)
    metadata = {
        "source_label": source_label,
        "source_is_read_only": True,
        "imgsz": int(imgsz),
        "split_order": list(SPLITS),
        "path_policy": "public outputs contain dataset-root-relative paths only",
        "quantile_method": "numpy.quantile(method='linear')",
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "resolved_root_count": len({str(root).casefold() for root in roots}),
    }
    return instances, audit, issues, metadata
