"""Reproducible D-FINE-N preparation and artifact helpers.

The detector implementation is never vendored here. Colab checks out the
official Peterande/D-FINE repository at a fixed commit and calls its
train.main function in the foreground. This module only owns dataset
conversion, transfer auditing, checkpoint mirroring, and result packaging.
"""

from __future__ import annotations

import ast
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

OFFICIAL_REPOSITORY = "https://github.com/Peterande/D-FINE.git"
OFFICIAL_COMMIT = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
OFFICIAL_CONFIG = "configs/dfine/custom/dfine_hgnetv2_n_custom.yml"
OFFICIAL_CHECKPOINT_URL = (
    "https://github.com/Peterande/storage/releases/download/"
    "dfinev1.0/dfine_n_coco.pth"
)
OFFICIAL_CHECKPOINT_BYTES = 15_489_558
OFFICIAL_CHECKPOINT_SHA256 = (
    "41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    """Atomically write a UTF-8 JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def _dataset_root(
    data_yaml: Path,
    payload: dict[str, Any],
    explicit_root: str | Path | None,
) -> Path:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(Path(explicit_root))
    configured = Path(str(payload.get("path", data_yaml.parent)))
    if configured.is_absolute():
        candidates.append(configured)
    else:
        candidates.append((data_yaml.parent / configured).resolve())
    candidates.append(data_yaml.parent)
    for candidate in dict.fromkeys(candidates):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"No dataset root exists among: {candidates}")


def _resolve_entries(root: Path, value: Any) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    output = []
    for item in values:
        path = Path(str(item))
        output.append(path if path.is_absolute() else root / path)
    return output


def _images_from_entry(entry: Path, root: Path) -> list[Path]:
    if entry.is_dir():
        return sorted(
            path.resolve()
            for path in entry.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
    if entry.is_file() and entry.suffix.lower() == ".txt":
        images = []
        for line in entry.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value:
                continue
            path = Path(value)
            images.append((path if path.is_absolute() else root / path).resolve())
        return images
    if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
        return [entry.resolve()]
    raise ValueError(f"Unsupported dataset split entry: {entry}")


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _copy_one(pair: tuple[Path, Path]) -> tuple[int, bool]:
    source, destination = pair
    size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        return size, False
    shutil.copyfile(source, destination)
    if destination.stat().st_size != size:
        raise IOError(f"Size mismatch after copy: {source} -> {destination}")
    return size, True


def copy_yolo_dataset_to_local(
    drive_data_yaml: str | Path,
    *,
    drive_data_root: str | Path | None,
    local_data_root: str | Path,
    local_data_yaml: str | Path,
    workers: int = 32,
) -> dict[str, Any]:
    """Copy a YOLO dataset from Drive with live file and byte progress."""

    source_yaml = Path(drive_data_yaml)
    if not source_yaml.is_file():
        raise FileNotFoundError(source_yaml)
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    for key in (*SPLITS, "names"):
        if key not in payload:
            raise KeyError(f"Dataset YAML is missing {key!r}")
    source_root = _dataset_root(source_yaml, payload, drive_data_root)
    destination_root = Path(local_data_root).resolve()

    print(f"扫描云盘数据集：{source_root}", flush=True)
    source_files: list[Path] = []
    for current_root, _directories, filenames in os.walk(source_root):
        current = Path(current_root)
        source_files.extend(current / name for name in filenames)
        if source_files and len(source_files) % 250 == 0:
            print(
                f"\r已发现 {len(source_files):,} 个文件……",
                end="",
                flush=True,
            )
    source_files.sort()
    if not source_files:
        raise FileNotFoundError(f"No files under {source_root}")
    print(f"\r共发现 {len(source_files):,} 个文件。          ", flush=True)

    jobs = [
        (source, destination_root / source.relative_to(source_root))
        for source in source_files
    ]
    copied_files = copied_bytes = processed_bytes = 0
    started = time.perf_counter()
    with (
        tqdm(
            total=len(jobs),
            unit="file",
            desc="数据集文件",
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stdout,
        ) as files_bar,
        tqdm(
            total=None,
            unit="B",
            unit_scale=True,
            desc="已处理字节",
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stdout,
        ) as bytes_bar,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {pool.submit(_copy_one, pair): pair[0] for pair in jobs}
        for future in concurrent.futures.as_completed(futures):
            size, copied = future.result()
            files_bar.update(1)
            bytes_bar.update(size)
            processed_bytes += size
            copied_files += int(copied)
            copied_bytes += size if copied else 0
            files_bar.set_postfix(
                copied=copied_files,
                workers=workers,
                GiB=f"{processed_bytes / 1024**3:.2f}",
                refresh=False,
            )

    local_payload = dict(payload)
    local_payload["path"] = str(destination_root)
    for split in SPLITS:
        rewritten = []
        for entry in _resolve_entries(source_root, payload[split]):
            try:
                relative = entry.resolve().relative_to(source_root)
            except ValueError as error:
                raise ValueError(
                    f"{split} entry is outside the copied root: {entry}"
                ) from error
            rewritten.append(relative.as_posix())
        local_payload[split] = (
            rewritten[0] if not isinstance(payload[split], list) else rewritten
        )
    local_yaml = Path(local_data_yaml)
    local_yaml.parent.mkdir(parents=True, exist_ok=True)
    local_yaml.write_text(
        yaml.safe_dump(local_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    report = {
        "source_yaml": str(source_yaml),
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "local_yaml": str(local_yaml),
        "files": len(source_files),
        "processed_bytes": processed_bytes,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        "数据集复制完成："
        f"{report['files']:,} 个文件，"
        f"{processed_bytes / 1024**3:.2f} GiB，"
        f"{report['elapsed_seconds']:.1f} 秒。",
        flush=True,
    )
    return report


def _class_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        keys = sorted(int(key) for key in value)
        if keys != list(range(len(keys))):
            raise ValueError(f"Class ids must be contiguous and zero-based: {keys}")
        return [str(value[key] if key in value else value[str(key)]) for key in keys]
    raise TypeError("Dataset names must be a list or mapping")


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, destination)
        return "copy"


def convert_yolo_to_dfine_coco(
    local_data_yaml: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Convert frozen YOLO HBB splits to zero-based COCO for D-FINE.

    D-FINE's official custom loader uses raw ids when category remapping is
    disabled. Therefore a one-class dataset must use category id 0.
    """

    source_yaml = Path(local_data_yaml)
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    root = _dataset_root(source_yaml, payload, None)
    names = _class_names(payload["names"])
    nc = int(payload.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"nc={nc} differs from len(names)={len(names)}")

    split_images: dict[str, list[Path]] = {}
    memberships: dict[Path, str] = {}
    for split in SPLITS:
        images = sorted(
            {
                image
                for entry in _resolve_entries(root, payload[split])
                for image in _images_from_entry(entry, root)
            }
        )
        if not images:
            raise ValueError(f"{split} has no images")
        for image in images:
            if not image.is_file():
                raise FileNotFoundError(image)
            previous = memberships.setdefault(image, split)
            if previous != split:
                raise ValueError(
                    f"Data leakage: {image} belongs to {previous} and {split}"
                )
        split_images[split] = images

    output = Path(output_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    categories = [
        {"id": class_id, "name": name, "supercategory": "ship"}
        for class_id, name in enumerate(names)
    ]
    report: dict[str, Any] = {
        "source_yaml": str(source_yaml),
        "source_root": str(root),
        "output_root": str(output),
        "category_id_policy": "zero_based_raw_ids_for_remap_false",
        "categories": categories,
        "splits": {},
    }
    used_labels: set[Path] = set()
    try:
        for split, images in split_images.items():
            coco_images = []
            annotations = []
            backgrounds = 0
            link_modes: dict[str, int] = {}
            for image_id, source in enumerate(
                tqdm(images, desc=f"转换 {split}", unit="image"),
                start=1,
            ):
                relative = source.relative_to(root).as_posix()
                digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
                file_name = f"{digest}_{source.name}"
                destination = temporary / "images" / split / file_name
                mode = _link_or_copy(source, destination)
                link_modes[mode] = link_modes.get(mode, 0) + 1
                with Image.open(source) as opened:
                    width, height = opened.size
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": file_name,
                        "width": width,
                        "height": height,
                    }
                )

                label = _label_for_image(source)
                used_labels.add(label.resolve())
                if not label.is_file():
                    raise FileNotFoundError(
                        f"Missing label for {source}: expected {label}"
                    )
                lines = [
                    line.strip()
                    for line in label.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if not lines:
                    backgrounds += 1
                for line_number, line in enumerate(lines, start=1):
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(
                            f"{label}:{line_number}: expected YOLO HBB 5 fields"
                        )
                    class_id = int(float(fields[0]))
                    if not 0 <= class_id < nc:
                        raise ValueError(
                            f"{label}:{line_number}: invalid class {class_id}"
                        )
                    cx, cy, box_width, box_height = map(float, fields[1:])
                    if not (
                        0 <= cx <= 1
                        and 0 <= cy <= 1
                        and 0 < box_width <= 1
                        and 0 < box_height <= 1
                    ):
                        raise ValueError(
                            f"{label}:{line_number}: invalid normalized box"
                        )
                    x1 = max(0.0, (cx - box_width / 2) * width)
                    y1 = max(0.0, (cy - box_height / 2) * height)
                    x2 = min(float(width), (cx + box_width / 2) * width)
                    y2 = min(float(height), (cy + box_height / 2) * height)
                    pixel_width = x2 - x1
                    pixel_height = y2 - y1
                    if pixel_width <= 0 or pixel_height <= 0:
                        raise ValueError(
                            f"{label}:{line_number}: degenerate clipped box"
                        )
                    annotations.append(
                        {
                            "id": len(annotations) + 1,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x1, y1, pixel_width, pixel_height],
                            "area": pixel_width * pixel_height,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )

            document = {
                "info": {
                    "description": "Frozen ship dataset converted from YOLO HBB",
                    "source_yaml": str(source_yaml),
                },
                "licenses": [],
                "images": coco_images,
                "annotations": annotations,
                "categories": categories,
            }
            annotation_path = temporary / "annotations" / f"instances_{split}.json"
            write_json(annotation_path, document)
            report["splits"][split] = {
                "images": len(coco_images),
                "instances": len(annotations),
                "background_images": backgrounds,
                "annotation_file": str(
                    output / "annotations" / annotation_path.name
                ),
                "annotation_sha256": sha256_file(annotation_path),
                "image_materialization": link_modes,
            }

        labels_root = root / "labels"
        all_labels = (
            {path.resolve() for path in labels_root.rglob("*.txt")}
            if labels_root.is_dir()
            else set()
        )
        report["unused_label_files"] = len(all_labels - used_labels)
        write_json(temporary / "conversion_report.json", report)
        if output.exists():
            if output == Path(output.anchor) or output.parent == output:
                raise RuntimeError(f"Refusing broad replacement: {output}")
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    print("✅ YOLO → D-FINE COCO 转换完成；train/val/test 保持冻结且互不重叠。")
    return report


def install_dfine_tuning_audit(audit_output: str | Path) -> None:
    """Wrap the official tuning loader with an exact tensor handoff audit."""

    from src.solver._solver import BaseSolver, remove_module_prefix

    if getattr(BaseSolver.load_tuning_state, "_ship_yolo_audited", False):
        return
    original = BaseSolver.load_tuning_state
    output = Path(audit_output)

    def audited(self, path: str) -> None:
        if str(path).startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(path, map_location="cpu")
            checkpoint_path = str(path)
            checkpoint_sha256 = None
        else:
            checkpoint_path = str(Path(path).resolve())
            try:
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            checkpoint_sha256 = sha256_file(path)
        pretrained = (
            checkpoint["ema"]["module"]
            if "ema" in checkpoint
            else checkpoint["model"]
        )
        pretrained = remove_module_prefix(pretrained)
        target_before = self.model.state_dict()
        compatible = {
            key: value
            for key, value in pretrained.items()
            if key in target_before
            and tuple(value.shape) == tuple(target_before[key].shape)
        }
        original(self, path)
        target_after = self.model.state_dict()
        failures = [
            key
            for key, value in compatible.items()
            if not torch.equal(target_after[key].detach().cpu(), value.detach().cpu())
        ]
        parameter_names = set(dict(self.model.named_parameters()))
        loaded_parameter_elements = sum(
            target_after[key].numel()
            for key in compatible
            if key in parameter_names
        )
        total_parameter_elements = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        report = {
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "policy": "official_tuning_loader_plus_exact_same_name_same_shape_audit",
            "source_tensors": len(pretrained),
            "target_tensors": len(target_before),
            "loaded_tensors": len(compatible),
            "loaded_total": f"{len(compatible)}/{len(target_before)}",
            "loaded_parameter_elements": loaded_parameter_elements,
            "target_parameter_elements": total_parameter_elements,
            "parameter_element_inheritance_ratio": (
                loaded_parameter_elements / total_parameter_elements
            ),
            "unmatched_or_missing_target_tensors": sorted(
                set(target_before) - set(compatible)
            ),
            "verification_failures": failures,
            "passed": bool(compatible) and not failures,
        }
        write_json(output, report)
        if not report["passed"]:
            raise RuntimeError(
                "D-FINE official checkpoint transfer audit failed: "
                + json.dumps(failures[:20])
            )
        print(
            "D-FINE 官方预训练权重 Loaded/Total："
            f"{report['loaded_total']}；参数元素继承率："
            f"{report['parameter_element_inheritance_ratio']:.2%}"
        )

    audited._ship_yolo_audited = True
    BaseSolver.load_tuning_state = audited


def parse_training_log(log_path: str | Path, *, stop_epoch: int) -> dict[str, Any]:
    """Select the best validation AP row from the official JSON-lines log."""

    rows = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = row.get("test_coco_eval_bbox")
        if isinstance(metrics, list) and len(metrics) >= 6:
            rows.append(row)
    if not rows:
        raise ValueError(f"No COCO validation rows in {log_path}")
    best = max(rows, key=lambda row: float(row["test_coco_eval_bbox"][0]))
    values = best["test_coco_eval_bbox"]
    epoch_zero_based = int(best["epoch"])
    checkpoint_name = (
        "best_stg1.pth" if epoch_zero_based < stop_epoch else "best_stg2.pth"
    )
    return {
        "best_epoch": epoch_zero_based + 1,
        "best_epoch_zero_based": epoch_zero_based,
        "checkpoint_name": checkpoint_name,
        "map50_95": float(values[0]),
        "map50": float(values[1]),
        "map75": float(values[2]),
        "map_small": float(values[3]),
        "map_medium": float(values[4]),
        "map_large": float(values[5]),
        "epochs_logged": len(rows),
        "parameters_reported": int(best["n_parameters"]),
    }


def parse_validator_metrics(text: str) -> dict[str, Any]:
    """Extract the final official Validator metric dictionary from stdout."""

    matches = re.findall(r"Metrics:\s*(\{[^\n]+\})", text)
    if not matches:
        raise ValueError("No 'Metrics: {...}' record found in validation output")
    value = ast.literal_eval(matches[-1])
    return {
        key: value[key]
        for key in ("precision", "recall", "f1", "iou", "TPs", "FPs", "FNs")
        if key in value
    }


def _atomic_copy(source: Path, destination: Path) -> bool:
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    shutil.copyfile(source, temporary)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        temporary.unlink(missing_ok=True)
        return False
    os.replace(temporary, destination)
    return True


@dataclass
class StableDriveMirror:
    """Mirror only stable artifacts while official training stays foreground."""

    local_dir: Path
    drive_dir: Path
    interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.local_dir = Path(self.local_dir)
        self.drive_dir = Path(self.drive_dir)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: dict[str, tuple[int, int]] = {}

    def _candidates(self) -> Iterable[Path]:
        names = (
            "log.txt",
            "last.pth",
            "best_stg1.pth",
            "best_stg2.pth",
            "pretrained_transfer_report.json",
            "runtime_config.yml",
            "dataset_conversion_report.json",
        )
        for name in names:
            path = self.local_dir / name
            if path.is_file():
                yield path

    def sync_once(self, *, require_stable: bool = True) -> list[str]:
        copied = []
        for source in self._candidates():
            relative = source.relative_to(self.local_dir).as_posix()
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            previous = self._seen.get(relative)
            self._seen[relative] = signature
            if require_stable and previous != signature:
                continue
            if _atomic_copy(source, self.drive_dir / relative):
                copied.append(relative)
        return copied

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            copied = self.sync_once(require_stable=True)
            if copied:
                print(f"\n☁️ 已同步稳定断点：{', '.join(copied)}", flush=True)

    def start(self) -> None:
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        self.sync_once(require_stable=True)
        self._thread = threading.Thread(
            target=self._run,
            name="dfine-drive-mirror",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds + 5.0))
        self.sync_once(require_stable=False)


def write_checksum_manifest(root: str | Path) -> Path:
    """Hash final files after all mutable state has stabilized."""

    directory = Path(root)
    output = directory / "artifact_checksums.sha256"
    rows = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path == output or path.name.startswith("."):
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output


def write_metrics_csv(path: str | Path, metrics: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return output
