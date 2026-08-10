"""Controlled Colab workflow for four remote-sensing ship paper reproductions."""

from __future__ import annotations

import concurrent.futures
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm

from custom_modules.register import register_custom_modules
from tools.external_baselines.ship_losses import get_loss_trainer


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments" / "paper_comparisons" / "protocol.yaml"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_protocol() -> dict[str, Any]:
    """Load the pinned comparison protocol."""

    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ComparisonRun:
    """Resolved paths and settings for one independent comparison run."""

    run_id: str
    method: str
    model_yaml: Path
    pretrained: str
    loss: str
    expected_strides: tuple[int, ...]
    run_name: str
    local_project: Path = Path("/content/comparison_runs")
    drive_project: Path = Path("/content/drive/MyDrive/ship_detection/paper_comparisons")

    @property
    def local_run(self) -> Path:
        return self.local_project / self.run_name

    @property
    def drive_run(self) -> Path:
        return self.drive_project / self.run_name


def resolve_run(run_id: str) -> ComparisonRun:
    """Resolve a registered run without guessing names or paths."""

    protocol = load_protocol()
    try:
        item = protocol["runs"][run_id]
    except KeyError as error:
        raise KeyError(f"Unknown comparison run ID: {run_id!r}.") from error
    return ComparisonRun(
        run_id=run_id,
        method=item["method"],
        model_yaml=ROOT / item["model_yaml"],
        pretrained=item["pretrained"],
        loss=item["loss"],
        expected_strides=tuple(int(value) for value in item["expected_strides"]),
        run_name=item["run_name"],
    )


def _scan_files(root: Path) -> list[Path]:
    files: list[Path] = []
    print(f"正在扫描云盘数据集：{root}", flush=True)
    for current, _directories, filenames in os.walk(root):
        current_path = Path(current)
        files.extend(current_path / filename for filename in filenames)
        if files and len(files) % 250 == 0:
            print(f"\r已发现 {len(files):,} 个文件……", end="", flush=True)
    files.sort()
    print(f"\r扫描完成，共 {len(files):,} 个文件。          ", flush=True)
    if not files:
        raise FileNotFoundError(f"云盘数据集为空或不可访问：{root}")
    return files


def _copy_one(source: Path, destination: Path) -> tuple[int, bool]:
    size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        return size, False
    shutil.copyfile(source, destination)
    if destination.stat().st_size != size:
        raise IOError(f"复制后大小不一致：{source} -> {destination}")
    return size, True


def copy_dataset_with_progress(
    source_root: str | Path = "/content/drive/MyDrive/ship_detection/data",
    destination_root: str | Path = "/content/ship_detection/data",
    workers: int = 32,
) -> dict[str, Any]:
    """Copy Drive data locally with real-time file and byte progress bars."""

    source_root = Path(source_root)
    destination_root = Path(destination_root)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            required = source_root / split / kind
            if not required.is_dir():
                raise FileNotFoundError(
                    f"缺少必需目录：{required}。程序没有猜测其他数据路径。"
                )

    source_files = _scan_files(source_root)
    jobs = [
        (source, destination_root / source.relative_to(source_root))
        for source in source_files
    ]
    processed_bytes = copied_bytes = copied_files = 0
    started = time.perf_counter()
    print(f"使用 {workers} 个线程复制到 Colab 本地：{destination_root}", flush=True)
    with (
        tqdm(
            total=len(jobs),
            desc="数据集文件",
            unit="file",
            dynamic_ncols=True,
            file=sys.stdout,
        ) as file_bar,
        tqdm(
            total=None,
            desc="已处理字节",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ) as byte_bar,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {
            pool.submit(_copy_one, source, destination): source
            for source, destination in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                size, copied = future.result()
            except Exception as error:
                raise IOError(f"复制失败：{source}；原因：{error}") from error
            processed_bytes += size
            copied_bytes += size if copied else 0
            copied_files += int(copied)
            file_bar.update(1)
            byte_bar.update(size)
            file_bar.set_postfix(
                新复制=copied_files,
                GiB=f"{processed_bytes / 1024**3:.2f}",
                refresh=False,
            )

    report = {
        "source": str(source_root),
        "destination": str(destination_root),
        "files": len(jobs),
        "processed_bytes": processed_bytes,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        "数据集本地复制完成："
        f"{len(jobs):,} 个文件，处理 {processed_bytes / 1024**3:.2f} GiB，"
        f"耗时 {report['elapsed_seconds']:.1f} 秒。",
        flush=True,
    )
    _write_json(destination_root.parent / "dataset_copy_report.json", report)
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_local_dataset(data_root: str | Path) -> dict[str, Any]:
    """Audit pairing, YOLO-HBB labels, class range and split leakage."""

    data_root = Path(data_root)
    report: dict[str, Any] = {
        "data_root": str(data_root),
        "read_only_source": True,
        "label_format": "class x_center y_center width height",
        "splits": {},
        "cross_split_duplicate_images": [],
    }
    split_hashes: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    for split in ("train", "val", "test"):
        image_dir = data_root / split / "images"
        label_dir = data_root / split / "labels"
        images = sorted(
            path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(label_dir.rglob("*.txt"))
        label_by_stem = {path.stem: path for path in labels}
        image_stems = {path.stem for path in images}
        missing_labels = [image.name for image in images if image.stem not in label_by_stem]
        extra_labels = [label.name for label in labels if label.stem not in image_stems]
        instances = empty_labels = invalid_rows = 0
        classes: set[int] = set()

        for image in images:
            label = label_by_stem.get(image.stem)
            if label is None:
                continue
            rows = [row.strip() for row in label.read_text(encoding="utf-8").splitlines() if row.strip()]
            if not rows:
                empty_labels += 1
            for line_number, row in enumerate(rows, start=1):
                fields = row.split()
                valid = len(fields) == 5
                try:
                    cls = int(fields[0]) if valid else -1
                    coordinates = [float(value) for value in fields[1:]] if valid else []
                    valid = (
                        valid
                        and cls == 0
                        and all(0.0 <= value <= 1.0 for value in coordinates)
                        and coordinates[2] > 0.0
                        and coordinates[3] > 0.0
                    )
                except (ValueError, IndexError):
                    valid = False
                    cls = -1
                if valid:
                    instances += 1
                    classes.add(cls)
                else:
                    invalid_rows += 1
                    if len(errors) < 30:
                        errors.append(f"{label}:{line_number}: {row}")

        if missing_labels:
            errors.append(f"{split} 缺少 {len(missing_labels)} 个图像标签")
        if invalid_rows:
            errors.append(f"{split} 包含 {invalid_rows} 行非法标签")
        if not images:
            errors.append(f"{split} 没有图像")
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": instances,
            "empty_labels": empty_labels,
            "missing_labels": len(missing_labels),
            "extra_labels": len(extra_labels),
            "extra_label_examples": extra_labels[:10],
            "invalid_rows": invalid_rows,
            "classes": sorted(classes),
        }
        split_hashes[split] = {image.name: _sha256(image) for image in images}

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        reverse = {digest: name for name, digest in split_hashes[right].items()}
        for left_name, digest in split_hashes[left].items():
            if digest in reverse:
                report["cross_split_duplicate_images"].append(
                    {
                        "splits": [left, right],
                        "left": left_name,
                        "right": reverse[digest],
                        "sha256": digest,
                    }
                )
    if report["cross_split_duplicate_images"]:
        errors.append(
            "检测到跨训练/验证/测试划分的相同图像，已拒绝训练以防泄漏"
        )

    report["errors"] = errors
    report_path = data_root.parent / "dataset_audit.json"
    _write_json(report_path, report)
    if errors:
        raise ValueError(
            "数据集审计未通过：\n- " + "\n- ".join(errors[:30])
        )
    print("数据集审计通过：", report["splits"], flush=True)
    return report


def prepare_local_dataset(
    source_root: str | Path = "/content/drive/MyDrive/ship_detection/data",
    destination_root: str | Path = "/content/ship_detection/data",
    workers: int = 32,
) -> tuple[Path, dict[str, Any]]:
    """Copy, audit and create a local-only dataset YAML."""

    destination_root = Path(destination_root)
    copy_dataset_with_progress(source_root, destination_root, workers=workers)
    audit = audit_local_dataset(destination_root)
    data_yaml = destination_root.parent / "data_runtime.yaml"
    payload = {
        "path": str(destination_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["ship"],
    }
    data_yaml.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print("已生成仅指向 Colab 本地副本的 data.yaml：", data_yaml, flush=True)
    return data_yaml, audit


def _matching_tensors(source: torch.nn.Module, target: torch.nn.Module) -> list[str]:
    source_state = source.state_dict()
    target_state = target.state_dict()
    return [
        name
        for name, tensor in target_state.items()
        if name in source_state and source_state[name].shape == tensor.shape
    ]


def build_initialized_model(run: ComparisonRun, nc: int = 1):
    """Build the paper graph, audit official weight inheritance and initialize it."""

    register_custom_modules()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    source = YOLO(run.pretrained).model
    target_for_audit = DetectionModel(str(run.model_yaml), nc=nc, verbose=False)
    matching = _matching_tensors(source, target_for_audit)
    total = len(target_for_audit.state_dict())
    if not matching:
        raise AssertionError(f"{run.run_id} 没有继承任何官方预训练张量。")

    model = YOLO(str(run.model_yaml), task="detect")
    model.load(run.pretrained)
    strides = tuple(int(value) for value in model.model.stride.tolist())
    if strides != run.expected_strides:
        raise AssertionError(
            f"检测步长不匹配：actual={strides}, expected={run.expected_strides}"
        )
    audit = {
        "run_id": run.run_id,
        "method": run.method,
        "model_yaml": str(run.model_yaml.relative_to(ROOT)),
        "pretrained": run.pretrained,
        "loaded_tensors": len(matching),
        "total_tensors": total,
        "loaded_total": f"{len(matching)}/{total}",
        "detect_strides": list(strides),
        "custom_loss": run.loss,
        "matching_tensor_examples": matching[:30],
    }
    print("官方预训练权重 Loaded/Total：", audit["loaded_total"], flush=True)
    print("检测步长：", audit["detect_strides"], flush=True)
    return model, audit


def cpu_smoke_test(model, imgsz: int = 128) -> None:
    """Run required CPU forward/backward without inference-mode tensor reuse."""

    core = model.model.cpu().train()
    sample = torch.randn(1, 3, imgsz, imgsz, requires_grad=True)
    output = core(sample)
    boxes = output["boxes"] if isinstance(output, dict) else output[0]
    loss = boxes.float().square().mean()
    loss.backward()
    if sample.grad is None or not torch.isfinite(sample.grad).all():
        raise AssertionError("CPU backward did not produce finite input gradients.")
    core.zero_grad(set_to_none=True)
    del sample, output, boxes, loss
    gc.collect()
    print("CPU 前向/反向检查：通过", flush=True)


def _copy_run_artifacts(source: Path, destination: Path) -> None:
    """Mirror recoverable run artifacts to Drive without deleting anything."""

    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not target.is_file() or target.stat().st_size != path.stat().st_size:
                shutil.copyfile(path, target)
        except OSError as error:
            print(f"警告：暂时无法同步 {relative}：{error}", flush=True)


def _mirror_callback(run: ComparisonRun):
    def callback(trainer) -> None:
        _copy_run_artifacts(Path(trainer.save_dir), run.drive_run)
        print(f"本轮训练产物已同步到云盘：{run.drive_run}", flush=True)

    return callback


def _find_checkpoint(run: ComparisonRun, filename: str) -> Path | None:
    for root in (run.local_run, run.drive_run):
        candidate = root / "weights" / filename
        if candidate.is_file():
            return candidate
    return None


def _checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1))


def train_foreground(run: ComparisonRun, data_yaml: Path):
    """Run or resume official foreground training with live epoch output."""

    protocol = load_protocol()
    settings = dict(protocol["training"])
    epochs = int(settings.pop("epochs"))
    local_last = _find_checkpoint(run, "last.pt")
    trainer_class = get_loss_trainer(run.loss)
    register_custom_modules()
    from ultralytics import YOLO

    if local_last is not None:
        last_epoch = _checkpoint_epoch(local_last)
        if last_epoch + 1 < epochs:
            print(
                f"发现可恢复断点：{local_last}（已完成 epoch {last_epoch + 1}/{epochs}）",
                flush=True,
            )
            model = YOLO(str(local_last))
            model.add_callback("on_fit_epoch_end", _mirror_callback(run))
            kwargs = {"resume": str(local_last)}
            if trainer_class is None:
                results = model.train(**kwargs)
            else:
                results = model.train(trainer=trainer_class, **kwargs)
            _copy_run_artifacts(Path(model.trainer.save_dir), run.drive_run)
            return model, results
        print("训练 checkpoint 已达到目标轮数，不重复训练。", flush=True)
        return YOLO(str(_find_checkpoint(run, "best.pt") or local_last)), None

    artifacts = []
    for root in (run.local_run, run.drive_run):
        if root.is_dir():
            artifacts.extend(path for path in root.rglob("*") if path.is_file())
    if artifacts:
        raise FileExistsError(
            f"实验目录存在产物但没有可恢复 last.pt：{run.drive_run}。"
            "程序不会删除或覆盖它，请先人工确认。"
        )

    model, transfer_audit = build_initialized_model(run, nc=1)
    cpu_smoke_test(model)
    run.local_run.mkdir(parents=True, exist_ok=True)
    _write_json(run.local_run / "pretrained_transfer_audit.json", transfer_audit)
    model.add_callback("on_fit_epoch_end", _mirror_callback(run))
    train_args = {
        "data": str(data_yaml),
        "project": str(run.local_project),
        "name": run.run_name,
        "exist_ok": False,
        "epochs": epochs,
        "imgsz": settings.pop("imgsz"),
        "batch": settings.pop("batch"),
        "workers": settings.pop("workers"),
        "device": 0,
        "val": True,
        "augment": True,
        **settings,
    }
    print("全部检查通过，开始当前内核中的官方 YOLO.train 前台训练。", flush=True)
    if trainer_class is None:
        results = model.train(**train_args)
    else:
        results = model.train(trainer=trainer_class, **train_args)
    _copy_run_artifacts(Path(model.trainer.save_dir), run.drive_run)
    return model, results


def _metrics_dict(metrics) -> dict[str, float]:
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "AP75": float(box.map75),
        "mAP50-95": float(box.map),
    }


def evaluate_best_on_val_and_test(run: ComparisonRun, data_yaml: Path) -> dict[str, Any]:
    """Evaluate the frozen best checkpoint on val and then the sealed test split."""

    register_custom_modules()
    from ultralytics import YOLO

    best = _find_checkpoint(run, "best.pt")
    if best is None:
        raise FileNotFoundError(f"找不到 {run.run_id} 的 best.pt。")
    model = YOLO(str(best))
    evaluation_root = run.local_run / "evaluation"
    summary: dict[str, Any] = {"run_id": run.run_id, "best_pt": str(best)}
    for split in ("val", "test"):
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=640,
            batch=8,
            workers=2,
            device=0,
            augment=False,
            plots=True,
            project=str(evaluation_root),
            name=split,
            exist_ok=True,
        )
        summary[split] = _metrics_dict(metrics)
        print(f"{split} 指标：", summary[split], flush=True)

    _write_json(run.local_run / "val_test_metrics.json", summary)
    with (run.local_run / "val_test_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["split", "precision", "recall", "mAP50", "AP75", "mAP50-95"],
        )
        writer.writeheader()
        for split in ("val", "test"):
            writer.writerow({"split": split, **summary[split]})
    _copy_run_artifacts(run.local_run, run.drive_run)
    print("验证集和测试集评估完成，产物已同步到：", run.drive_run, flush=True)
    return summary
