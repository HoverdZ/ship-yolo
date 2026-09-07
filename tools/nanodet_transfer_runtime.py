"""Foreground Colab runtime for the paired NanoDet-Plus ship experiments.

This file deliberately avoids PyTorch Lightning.  NanoDet v1.0.0 pins an old
Lightning/PyTorch stack that is incompatible with current Colab Python, while
its detector, dataset, augmentation, loss, evaluator, and EMA code are usable
with the preinstalled current PyTorch stack.  Training therefore runs in the
notebook process through a small native PyTorch loop; no training subprocess
is created and every batch/epoch remains visible in real time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

NANODET_COMMIT = "d3fb34fa91d6020f273d6d063bf324dcd97bac12"
NANODET_WEIGHT_ID = "1FN3WK3FLjBm7oCqiwUcD3m3MjfqxuzXe"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

_MINIMAL_NANODET_UTIL = '''"""Current-Colab import shim for NanoDet v1.0.0.

Only the utilities required by the official model/head code are exported.
The removed exports pull in legacy PyTorch Lightning, which is not used by
the foreground training loop and is incompatible with current Colab Python.
"""
from .box_transform import bbox2distance, distance2bbox
from .misc import images_to_levels, multi_apply, unmap
from . import util_mixins

def overlay_bbox_cv(*args, **kwargs):
    # Visualization is not needed for training; import it only if explicitly used.
    from .visualization import overlay_bbox_cv as implementation
    return implementation(*args, **kwargs)

__all__ = [
    "bbox2distance",
    "distance2bbox",
    "images_to_levels",
    "multi_apply",
    "overlay_bbox_cv",
    "unmap",
    "util_mixins",
]
'''


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def activate_nanodet_source(source_root: str | Path) -> dict[str, Any]:
    """Activate the pinned official source without changing binary packages."""

    source = Path(source_root).expanduser().resolve()
    required = (
        source / "nanodet/model/arch/nanodet_plus.py",
        source / "nanodet/model/head/nanodet_plus_head.py",
        source / "nanodet/data/dataset/coco.py",
        source / "nanodet/evaluator/coco_detection.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "NanoDet v1.0.0 source is incomplete: " + ", ".join(missing)
        )
    if any(name == "nanodet.util" or name.startswith("nanodet.model") for name in sys.modules):
        raise RuntimeError(
            "NanoDet was imported before the compatibility shim. Restart the "
            "runtime and run the notebook cells in order."
        )

    util_init = source / "nanodet/util/__init__.py"
    util_init.write_text(_MINIMAL_NANODET_UTIL, encoding="utf-8")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    # Import every expensive-run dependency now, before Drive data is copied.
    import cv2
    import pycocotools
    import torchvision
    from nanodet.data.dataset.coco import CocoDataset  # noqa: F401
    from nanodet.evaluator.coco_detection import CocoDetectionEvaluator  # noqa: F401
    from nanodet.model.head.nanodet_plus_head import NanoDetPlusHead  # noqa: F401

    report = {
        "nanodet_source": str(source),
        "nanodet_commit": NANODET_COMMIT,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pycocotools": getattr(pycocotools, "__version__", "installed"),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def load_protocol(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise TypeError(f"Protocol must be a mapping: {path}")
    required = {
        "experiment",
        "variant",
        "input_size",
        "batch_size",
        "epochs",
        "seed",
        "optimizer",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise KeyError(f"Protocol is missing keys: {missing}")
    if protocol["variant"] not in {"official", "ours"}:
        raise ValueError("variant must be 'official' or 'ours'.")
    if int(protocol["input_size"]) != 416:
        raise ValueError("NanoDet-Plus-m transfer protocol is fixed at 416 input.")
    if int(protocol["seed"]) != 0:
        raise ValueError("The controlled pair is fixed at seed=0.")
    return protocol


def download_official_checkpoint(
    destination: str | Path,
    *,
    file_id: str = NANODET_WEIGHT_ID,
) -> Path:
    """Download the official NanoDet-Plus-m-416 detector weights once."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1_000_000:
        print(f"复用官方预训练权重：{path}", flush=True)
        return path
    if path.exists():
        path.unlink()
    import gdown

    print("正在下载官方 NanoDet-Plus-m-416 权重……", flush=True)
    result = gdown.download(id=file_id, output=str(path), quiet=False)
    if result is None or not path.is_file() or path.stat().st_size <= 1_000_000:
        raise RuntimeError(
            "官方 NanoDet 权重下载失败；数据集尚未复制，因此不会浪费云盘 I/O。"
        )
    return path


@dataclass
class PreparedNanoDetExperiment:
    protocol_path: Path
    protocol: dict[str, Any]
    model: nn.Module
    checkpoint_path: Path
    environment: dict[str, Any]
    structure: dict[str, Any]
    transfer: dict[str, Any]


def prepare_nanodet_experiment(
    config_path: str | Path,
    *,
    source_root: str | Path,
    checkpoint_path: str | Path,
) -> PreparedNanoDetExperiment:
    """Validate environment, structure, and transfer before dataset copying."""

    environment = activate_nanodet_source(source_root)
    protocol_path = Path(config_path).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    set_reproducibility(int(protocol["seed"]))
    checkpoint = download_official_checkpoint(checkpoint_path)

    # Import only after the pinned source and compatibility layer are active.
    from custom_modules.nanodet_transfer import (
        audit_nanodet_pair_structure,
        build_nanodet_plus_pair_model,
        load_audited_nanodet_checkpoint,
    )

    model = build_nanodet_plus_pair_model(protocol["variant"], num_classes=1)
    structure = audit_nanodet_pair_structure(
        model,
        variant=protocol["variant"],
        input_size=64,
    )
    transfer = load_audited_nanodet_checkpoint(
        model,
        checkpoint,
        variant=protocol["variant"],
    )
    print(
        f"结构检查通过：strides={structure['feature_strides']}，"
        f"params={structure['parameters']:,}",
        flush=True,
    )
    return PreparedNanoDetExperiment(
        protocol_path=protocol_path,
        protocol=protocol,
        model=model,
        checkpoint_path=checkpoint,
        environment=environment,
        structure=structure,
        transfer=transfer,
    )


def _collect_copy_jobs(source_root: Path, destination_root: Path) -> list[tuple[Path, Path, int]]:
    jobs: list[tuple[Path, Path, int]] = []
    copied_labels: set[Path] = set()
    for split in ("train", "val", "test"):
        image_root = source_root / split / "images"
        label_root = source_root / split / "labels"
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(
                f"数据集必须包含 {split}/images 和 {split}/labels：{source_root}"
            )
        images = sorted(
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise RuntimeError(f"{image_root} 中没有支持的图像。")
        for image in images:
            relative = image.relative_to(source_root)
            jobs.append((image, destination_root / relative, image.stat().st_size))
            image_relative = image.relative_to(image_root)
            label = label_root / image_relative.with_suffix(".txt")
            if label.is_file() and label not in copied_labels:
                label_relative = Path(split) / "labels" / image_relative.with_suffix(".txt")
                jobs.append(
                    (label, destination_root / label_relative, label.stat().st_size)
                )
                copied_labels.add(label)
    return jobs


def _copy_one(job: tuple[Path, Path, int]) -> int:
    source, destination, size = job
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return size


def copy_dataset_with_live_progress(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    workers: int = 32,
) -> dict[str, Any]:
    """Copy matching YOLO files with simultaneous file and byte progress."""

    source = Path(source_root).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    marker = destination / ".nanodet_copy_complete.json"
    if marker.is_file():
        report = json.loads(marker.read_text(encoding="utf-8"))
        print(
            f"复用本地数据集：{report['files']} 个文件，"
            f"{report['bytes'] / 1024**3:.2f} GiB",
            flush=True,
        )
        return report
    if not source.is_dir():
        raise FileNotFoundError(f"Google Drive 数据集不存在：{source}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    print("正在枚举云盘图像及其匹配标签……", flush=True)
    jobs = _collect_copy_jobs(source, destination)
    total_bytes = sum(job[2] for job in jobs)
    started = time.time()
    with tqdm(
        total=len(jobs),
        desc="复制文件",
        unit="file",
        dynamic_ncols=True,
        position=0,
    ) as file_bar, tqdm(
        total=total_bytes,
        desc="复制字节",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        position=1,
    ) as byte_bar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_copy_one, job) for job in jobs]
            for future in as_completed(futures):
                copied = future.result()
                file_bar.update(1)
                byte_bar.update(copied)

    report = {
        "source": str(source),
        "destination": str(destination),
        "files": len(jobs),
        "bytes": total_bytes,
        "seconds": time.time() - started,
        "workers": workers,
    }
    _json_dump(marker, report)
    print(
        f"复制完成：{len(jobs)} 个文件，{total_bytes / 1024**3:.2f} GiB，"
        f"耗时 {report['seconds']:.1f} 秒。",
        flush=True,
    )
    return report


def _read_yolo_image_record(
    image_path: Path,
    image_root: Path,
    label_root: Path,
) -> tuple[str, int, int, list[list[float]]]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像：{image_path}")
    height, width = image.shape[:2]
    relative = image_path.relative_to(image_root)
    label_path = label_root / relative.with_suffix(".txt")
    boxes: list[list[float]] = []
    if label_path.is_file():
        for line_number, raw in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise ValueError(
                    f"标签必须为 5 列 YOLO HBB：{label_path}:{line_number}"
                )
            class_value, x_center, y_center, box_width, box_height = map(float, fields)
            if int(class_value) != 0 or class_value != int(class_value):
                raise ValueError(
                    f"仅允许类别 0（ship）：{label_path}:{line_number}"
                )
            if not all(
                math.isfinite(value)
                for value in (x_center, y_center, box_width, box_height)
            ):
                raise ValueError(f"标签含非有限值：{label_path}:{line_number}")
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"标签宽高必须为正：{label_path}:{line_number}")
            x1 = max(0.0, (x_center - box_width / 2.0) * width)
            y1 = max(0.0, (y_center - box_height / 2.0) * height)
            x2 = min(float(width), (x_center + box_width / 2.0) * width)
            y2 = min(float(height), (y_center + box_height / 2.0) * height)
            clipped_width = x2 - x1
            clipped_height = y2 - y1
            if clipped_width < 1.0 or clipped_height < 1.0:
                continue
            boxes.append([x1, y1, clipped_width, clipped_height])
    return relative.as_posix(), width, height, boxes


def generate_data_yaml_and_coco(
    local_data_root: str | Path,
    coco_root: str | Path,
    *,
    workers: int = 16,
) -> dict[str, Any]:
    """Generate local data.yaml and deterministic COCO annotations."""

    data_root = Path(local_data_root).expanduser().resolve()
    coco = Path(coco_root).expanduser().resolve()
    annotations_root = coco / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)
    data_yaml = data_root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(data_root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "ship"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {"data_yaml": str(data_yaml), "splits": {}}
    for split in ("train", "val", "test"):
        image_root = data_root / split / "images"
        label_root = data_root / split / "labels"
        images = sorted(
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        records: list[tuple[str, int, int, list[list[float]]] | None] = [None] * len(images)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    _read_yolo_image_record,
                    image,
                    image_root,
                    label_root,
                ): index
                for index, image in enumerate(images)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc=f"生成 {split} COCO",
                unit="image",
                dynamic_ncols=True,
            ):
                records[future_to_index[future]] = future.result()

        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []
        annotation_id = 1
        for image_id, record in enumerate(records, start=1):
            if record is None:
                raise AssertionError("COCO conversion worker returned no record.")
            file_name, width, height, boxes = record
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
            for box in boxes:
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                annotation_id += 1
        payload = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "ship", "supercategory": "ship"}],
        }
        annotation_path = annotations_root / f"instances_{split}.json"
        _json_dump(annotation_path, payload)
        summary["splits"][split] = {
            "images": len(coco_images),
            "instances": len(coco_annotations),
            "image_root": str(image_root),
            "annotation": str(annotation_path),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def prepare_ship_dataset(
    drive_data_root: str | Path,
    local_data_root: str | Path = "/content/ship_detection/data",
    coco_root: str | Path = "/content/ship_detection/nanodet_coco",
) -> dict[str, Any]:
    copy_report = copy_dataset_with_live_progress(
        drive_data_root,
        local_data_root,
        workers=32,
    )
    conversion = generate_data_yaml_and_coco(
        local_data_root,
        coco_root,
        workers=16,
    )
    return {"copy": copy_report, "conversion": conversion}


def _naive_collate(batch: Sequence[Any]) -> Any:
    element = batch[0]
    if isinstance(element, Mapping):
        return {key: _naive_collate([item[key] for item in batch]) for key in element}
    return list(batch)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_datasets(
    data_root: Path,
    coco_root: Path,
    *,
    input_size: int,
) -> tuple[Any, Any]:
    from nanodet.data.dataset.coco import CocoDataset

    train_pipeline = {
        "perspective": 0.0,
        "scale": [0.6, 1.4],
        "stretch": [[0.8, 1.2], [0.8, 1.2]],
        "rotation": 0,
        "shear": 0,
        "translate": 0.2,
        "flip": 0.5,
        "brightness": 0.2,
        "contrast": [0.6, 1.4],
        "saturation": [0.5, 1.2],
        "normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]],
    }
    evaluation_pipeline = {
        "normalize": [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
    }
    train_dataset = CocoDataset(
        img_path=str(data_root / "train/images"),
        ann_path=str(coco_root / "annotations/instances_train.json"),
        input_size=(input_size, input_size),
        pipeline=train_pipeline,
        keep_ratio=False,
        mode="train",
    )
    val_dataset = CocoDataset(
        img_path=str(data_root / "val/images"),
        ann_path=str(coco_root / "annotations/instances_val.json"),
        input_size=(input_size, input_size),
        pipeline=evaluation_pipeline,
        keep_ratio=False,
        mode="val",
    )
    return train_dataset, val_dataset


def _build_test_dataset(data_root: Path, coco_root: Path, *, input_size: int) -> Any:
    from nanodet.data.dataset.coco import CocoDataset

    return CocoDataset(
        img_path=str(data_root / "test/images"),
        ann_path=str(coco_root / "annotations/instances_test.json"),
        input_size=(input_size, input_size),
        pipeline={
            "normalize": [
                [103.53, 116.28, 123.675],
                [57.375, 57.12, 58.395],
            ]
        },
        keep_ratio=False,
        mode="test",
    )


def _stack_and_move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    images = torch.stack(batch["img"], dim=0).contiguous()
    batch["img"] = images.to(device=device, non_blocking=True)
    return batch


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=_naive_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _learning_rate(
    global_step: int,
    total_steps: int,
    *,
    base_lr: float,
    minimum_lr: float,
    warmup_steps: int,
    warmup_ratio: float,
) -> float:
    if global_step < warmup_steps:
        alpha = global_step / max(warmup_steps, 1)
        return base_lr * (warmup_ratio + (1.0 - warmup_ratio) * alpha)
    progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum_lr + 0.5 * (base_lr - minimum_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _sync_artifacts(local_root: Path, drive_root: Path) -> None:
    allowed = {".pt", ".csv", ".json", ".yaml", ".yml", ".log", ".txt"}
    for source in local_root.rglob("*"):
        if source.is_file() and source.suffix.lower() in allowed:
            _atomic_copy(source, drive_root / source.relative_to(local_root))


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    evaluator: Any,
    output_dir: Path,
    device: torch.device,
    *,
    description: str,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    results: dict[int, Any] = {}
    for batch in tqdm(loader, desc=description, unit="batch", dynamic_ncols=True):
        batch = _stack_and_move(batch, device)
        predictions = model(batch["img"])
        detections = model.head.post_process(predictions, batch)
        results.update(detections)
    metrics = evaluator.evaluate(results, str(output_dir), rank=-1)
    return {key: float(value) for key, value in metrics.items()}


def _build_optimizer(model: nn.Module, optimizer_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(optimizer_cfg["name"])
    if name != "AdamW":
        raise ValueError("The paired NanoDet protocol uses official AdamW only.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )


def _restore_last_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Any,
    scaler: Any,
    variant: str,
) -> tuple[int, int, float]:
    if not path.is_file():
        return 0, 0, -1.0
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("variant") != variant:
        raise RuntimeError(
            f"续训目录属于 {checkpoint.get('variant')!r}，当前为 {variant!r}。"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    ema.load_state_dict(checkpoint["ema"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint["global_step"])
    best_map = float(checkpoint.get("best_map", -1.0))
    print(
        f"从最近完整轮次续训：epoch {start_epoch}/{checkpoint['epochs']}，"
        f"best mAP={best_map:.4f}",
        flush=True,
    )
    return start_epoch, global_step, best_map


def run_nanodet_training(
    prepared: PreparedNanoDetExperiment,
    *,
    local_data_root: str | Path = "/content/ship_detection/data",
    coco_root: str | Path = "/content/ship_detection/nanodet_coco",
    local_runs_root: str | Path = "/content/formal_runs",
    drive_runs_root: str | Path = "/content/drive/MyDrive/ship_detection/nanodet_transfer_runs",
) -> dict[str, Any]:
    """Run training directly in the active notebook process and final test."""

    if not torch.cuda.is_available():
        raise RuntimeError("正式训练需要 Colab GPU；当前未检测到 CUDA。")
    protocol = prepared.protocol
    experiment = str(protocol["experiment"])
    variant = str(protocol["variant"])
    seed = int(protocol["seed"])
    set_reproducibility(seed)

    data_root = Path(local_data_root).expanduser().resolve()
    coco = Path(coco_root).expanduser().resolve()
    local_output = Path(local_runs_root).expanduser().resolve() / experiment / "seed_0"
    drive_output = Path(drive_runs_root).expanduser().resolve() / experiment / "seed_0"
    local_output.mkdir(parents=True, exist_ok=True)
    drive_output.mkdir(parents=True, exist_ok=True)
    _json_dump(local_output / "environment.json", prepared.environment)
    _json_dump(local_output / "structure_audit.json", prepared.structure)
    _json_dump(local_output / "pretrained_transfer.json", prepared.transfer)
    shutil.copyfile(prepared.protocol_path, local_output / "protocol.yaml")

    input_size = int(protocol["input_size"])
    batch_size = int(protocol["batch_size"])
    workers = int(protocol["workers"])
    epochs = int(protocol["epochs"])
    val_interval = int(protocol["validation_interval"])
    train_dataset, val_dataset = _build_datasets(
        data_root,
        coco,
        input_size=input_size,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
        seed=seed,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
    )
    from nanodet.evaluator.coco_detection import CocoDetectionEvaluator
    from nanodet.model.weight_averager.ema import ExpMovingAverager

    evaluator = CocoDetectionEvaluator(val_dataset)
    device = torch.device("cuda:0")
    model = prepared.model.to(device)
    optimizer = _build_optimizer(model, protocol["optimizer"])
    amp_enabled = bool(protocol["amp"])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = ExpMovingAverager(decay=float(protocol["ema_decay"]), device=device)
    ema.load_from(model)

    local_last = local_output / "last.pt"
    drive_last = drive_output / "last.pt"
    if not local_last.is_file() and drive_last.is_file():
        _atomic_copy(drive_last, local_last)
        for name in ("best.pt", "results.csv"):
            drive_artifact = drive_output / name
            local_artifact = local_output / name
            if drive_artifact.is_file() and not local_artifact.is_file():
                _atomic_copy(drive_artifact, local_artifact)
    start_epoch, global_step, best_map = _restore_last_checkpoint(
        local_last,
        model=model,
        optimizer=optimizer,
        ema=ema,
        scaler=scaler,
        variant=variant,
    )

    total_steps = epochs * len(train_loader)
    base_lr = float(protocol["optimizer"]["lr"])
    minimum_lr = float(protocol["optimizer"]["eta_min"])
    warmup_steps = int(protocol["warmup_steps"])
    warmup_ratio = float(protocol["warmup_ratio"])
    grad_clip = float(protocol["grad_clip"])
    results_csv = local_output / "results.csv"
    best_path = local_output / "best.pt"

    print("=" * 88, flush=True)
    print(f"实验：{experiment} ({variant})", flush=True)
    print(
        f"416 输入 | batch={batch_size} | epochs={epochs} | seed={seed} | "
        f"AdamW lr={base_lr:g} | AMP={amp_enabled}",
        flush=True,
    )
    print(f"本地输出：{local_output}", flush=True)
    print(f"云盘输出：{drive_output}", flush=True)
    print("训练在当前 Notebook 进程前台直接运行。", flush=True)
    print("=" * 88, flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        model.set_epoch(epoch)
        sums: dict[str, float] = {}
        seen = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            batch = _stack_and_move(batch, device)
            lr = _learning_rate(
                global_step,
                total_steps,
                base_lr=base_lr,
                minimum_lr=minimum_lr,
                warmup_steps=warmup_steps,
                warmup_ratio=warmup_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                _, loss, loss_states = model.forward_train(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"epoch {epoch + 1} step {global_step}: loss={loss.item()}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model, global_step)
            global_step += 1

            batch_size_actual = int(batch["img"].shape[0])
            seen += batch_size_actual
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach()) * batch_size_actual
            for name, value in loss_states.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_size_actual
            progress.set_postfix(
                loss=f"{sums['loss'] / seen:.4f}",
                lr=f"{lr:.2e}",
                mem=f"{torch.cuda.memory_reserved() / 1024**3:.1f}G",
            )

        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": sums.get("loss", 0.0) / max(seen, 1),
            "loss_qfl": sums.get("loss_qfl", 0.0) / max(seen, 1),
            "loss_bbox": sums.get("loss_bbox", 0.0) / max(seen, 1),
            "loss_dfl": sums.get("loss_dfl", 0.0) / max(seen, 1),
            "aux_loss_qfl": sums.get("aux_loss_qfl", 0.0) / max(seen, 1),
            "aux_loss_bbox": sums.get("aux_loss_bbox", 0.0) / max(seen, 1),
            "aux_loss_dfl": sums.get("aux_loss_dfl", 0.0) / max(seen, 1),
            "mAP": "",
            "AP_50": "",
            "AP_75": "",
            "AP_small": "",
            "AP_m": "",
            "AP_l": "",
        }

        should_validate = (epoch + 1) % val_interval == 0 or epoch + 1 == epochs
        if should_validate:
            evaluation_model = copy_model_with_ema(model, ema, device)
            metrics = _evaluate(
                evaluation_model,
                val_loader,
                evaluator,
                local_output,
                device,
                description=f"Val {epoch + 1}/{epochs}",
            )
            del evaluation_model
            torch.cuda.empty_cache()
            row.update(metrics)
            current_map = metrics["mAP"]
            print(
                f"Val epoch {epoch + 1}: mAP={current_map:.4f}, "
                f"AP50={metrics['AP_50']:.4f}, AP75={metrics['AP_75']:.4f}",
                flush=True,
            )
            if current_map > best_map:
                best_map = current_map
                _atomic_torch_save(
                    {
                        "state_dict": ema.state_dict(),
                        "epoch": epoch,
                        "variant": variant,
                        "metrics": metrics,
                        "source_checkpoint_sha256": prepared.transfer[
                            "checkpoint_sha256"
                        ],
                    },
                    best_path,
                )

        _append_csv(results_csv, row)
        last_payload = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "epochs": epochs,
            "global_step": global_step,
            "best_map": best_map,
            "variant": variant,
            "experiment": experiment,
        }
        _atomic_torch_save(last_payload, local_last)
        _sync_artifacts(local_output, drive_output)
        print(
            f"Epoch {epoch + 1}/{epochs} 完成；当前最佳 mAP={best_map:.4f}；"
            "检查点已同步到云盘。",
            flush=True,
        )

    if not best_path.is_file() and (drive_output / "best.pt").is_file():
        _atomic_copy(drive_output / "best.pt", best_path)
    if not best_path.is_file():
        raise RuntimeError("训练结束但没有生成 best.pt。")

    # Test remains sealed until training has ended and best.pt is fixed.
    try:
        best_document = torch.load(best_path, map_location="cpu", weights_only=False)
    except TypeError:
        best_document = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_document["state_dict"], strict=True)
    model.to(device).eval()
    test_dataset = _build_test_dataset(data_root, coco, input_size=input_size)
    test_loader = _make_loader(
        test_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
    )
    test_evaluator = CocoDetectionEvaluator(test_dataset)
    test_metrics = _evaluate(
        model,
        test_loader,
        test_evaluator,
        local_output / "test",
        device,
        description="Final sealed test",
    )
    final = {
        "experiment": experiment,
        "variant": variant,
        "best_epoch": int(best_document["epoch"]) + 1,
        "validation_metrics": best_document["metrics"],
        "test_metrics": test_metrics,
        "best_checkpoint": str(drive_output / "best.pt"),
        "results_csv": str(drive_output / "results.csv"),
    }
    _json_dump(local_output / "final_metrics.json", final)
    _sync_artifacts(local_output, drive_output)
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
    return final


def copy_model_with_ema(model: nn.Module, ema: Any, device: torch.device) -> nn.Module:
    """Create an evaluation-only model without mutating the training weights."""

    import copy

    evaluation_model = copy.deepcopy(model).to(device)
    ema.apply_to(evaluation_model)
    evaluation_model.eval()
    return evaluation_model


__all__ = [
    "NANODET_COMMIT",
    "NANODET_WEIGHT_ID",
    "PreparedNanoDetExperiment",
    "activate_nanodet_source",
    "copy_dataset_with_live_progress",
    "download_official_checkpoint",
    "generate_data_yaml_and_coco",
    "load_protocol",
    "prepare_nanodet_experiment",
    "prepare_ship_dataset",
    "run_nanodet_training",
]
