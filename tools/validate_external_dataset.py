"""Read-only validation for a future second ship-detection dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_DESCRIPTOR_FIELDS = {
    "dataset_name",
    "root",
    "train",
    "val",
    "test",
    "class_mapping",
    "annotation_format",
    "license",
    "citation",
    "image_resolution",
    "notes",
}


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: str | Path) -> dict[str, Any]:
    descriptor_path = Path(path)
    payload = yaml.safe_load(descriptor_path.read_text(encoding="utf-8")) or {}
    missing_fields = sorted(REQUIRED_DESCRIPTOR_FIELDS - set(payload))
    null_fields = sorted(
        field for field in REQUIRED_DESCRIPTOR_FIELDS if payload.get(field) is None
    )
    if missing_fields or null_fields:
        raise ValueError(
            f"Incomplete descriptor; missing={missing_fields}, null={null_fields}"
        )
    if str(payload["annotation_format"]).lower() not in {
        "yolo",
        "yolo_detection",
    }:
        raise NotImplementedError(
            "This validator does not silently convert annotations. Convert the "
            "declared format with a separate audited manifest first."
        )
    root = Path(payload["root"]).expanduser().resolve()
    splits: dict[str, Any] = {}
    hashes: dict[str, list[str]] = {}
    class_ids: set[int] = set()
    for split in ("train", "val", "test"):
        split_path = Path(payload[split])
        split_path = split_path if split_path.is_absolute() else root / split_path
        images = sorted(
            item for item in split_path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = [_label_path(image) for image in images]
        corrupt = []
        missing_labels = []
        empty_labels = 0
        invalid_rows = []
        instances = 0
        split_hashes = []
        for image, label in zip(images, labels, strict=True):
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except Exception as error:
                corrupt.append({"image": str(image), "error": str(error)})
            split_hashes.append(_sha(image))
            if not label.is_file():
                missing_labels.append(str(label))
                continue
            lines = [
                line.strip()
                for line in label.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_labels += 1
            for number, line in enumerate(lines, start=1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:5]]
                    valid = (
                        len(fields) == 5
                        and all(value == value for value in values)
                        and all(0.0 <= value <= 1.0 for value in values)
                        and values[2] > 0.0
                        and values[3] > 0.0
                    )
                except (ValueError, IndexError):
                    class_id = -1
                    valid = False
                if valid:
                    class_ids.add(class_id)
                    instances += 1
                else:
                    invalid_rows.append(
                        {"label": str(label), "line": number, "text": line}
                    )
        hashes[split] = split_hashes
        splits[split] = {
            "path": str(split_path),
            "images": len(images),
            "labels": sum(label.is_file() for label in labels),
            "instances": instances,
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
            "invalid_rows": invalid_rows,
            "corrupt_images": corrupt,
        }
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_{right}"] = sorted(
            set(hashes[left]).intersection(hashes[right])
        )
    mapping_ids = {int(key) for key in payload["class_mapping"]}
    report = {
        "descriptor": str(descriptor_path),
        "dataset_name": payload["dataset_name"],
        "read_only": True,
        "license": payload["license"],
        "citation": payload["citation"],
        "annotation_format": payload["annotation_format"],
        "splits": splits,
        "observed_class_ids": sorted(class_ids),
        "declared_class_ids": sorted(mapping_ids),
        "class_mapping_matches": class_ids.issubset(mapping_ids),
        "exact_image_hash_overlap": overlaps,
    }
    errors = []
    for split, values in splits.items():
        if not values["images"]:
            errors.append(f"{split}: no images")
        for field in ("missing_labels", "invalid_rows", "corrupt_images"):
            if values[field]:
                errors.append(f"{split}: {len(values[field])} {field}")
    if not report["class_mapping_matches"]:
        errors.append("observed class IDs are outside class_mapping")
    if any(overlaps.values()):
        errors.append("exact duplicate images cross train/val/test splits")
    report["errors"] = errors
    report["passed"] = not errors
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate(args.descriptor)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = [
            "split",
            "images",
            "labels",
            "instances",
            "empty_labels",
            "missing_labels",
            "invalid_rows",
            "corrupt_images",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, values in report["splits"].items():
            writer.writerow(
                {
                    "split": split,
                    **{
                        key: len(values[key])
                        if isinstance(values[key], list)
                        else values[key]
                        for key in fields[1:]
                    },
                }
            )
    if not report["passed"]:
        raise SystemExit("External dataset validation failed; inspect the JSON report.")
    print(output)


if __name__ == "__main__":
    main()
