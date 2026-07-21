"""Shared direct-process Colab training workflow for the two FaPN variants."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fapn_utils import (
    EXPECTED_ULTRALYTICS_VERSION,
    build_model,
    model_statistics,
    require_ultralytics_version,
    semantic_weight_transfer,
    structure_report,
    variant_config,
    write_json,
)


METADATA_FILE = "experiment_metadata.json"
TRANSFER_FILE = "weight_transfer.json"
DATASET_AUDIT_FILE = "dataset_audit.json"
STAGING_FILES = {METADATA_FILE, TRANSFER_FILE, DATASET_AUDIT_FILE}
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def git_commit() -> str:
    """Resolve HEAD without launching another process."""

    git_path = ROOT / ".git"
    try:
        if git_path.is_file():
            value = git_path.read_text(encoding="utf-8").split(":", 1)[1].strip()
            git_dir = Path(value) if Path(value).is_absolute() else (ROOT / value).resolve()
        else:
            git_dir = git_path
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref_name = head.removeprefix("ref: ")
        loose_ref = git_dir / ref_name
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="utf-8").strip()
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref_name:
                    return commit
    except (OSError, IndexError, ValueError):
        pass
    return "unknown"


def validate_resume_checkpoint(path: str | Path) -> Path:
    import torch

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid resume checkpoint type: {type(payload).__name__}")
    missing = [
        key for key in ("epoch", "optimizer", "train_args") if key not in payload or payload[key] is None
    ]
    if missing:
        raise RuntimeError(f"Invalid resume checkpoint {checkpoint}; missing {missing}.")
    return checkpoint


def prepare_run_directory(run_dir: Path) -> None:
    """Prevent accidental overwrite while allowing this script's staging files."""

    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"Run path is not a directory: {run_dir}")
    entries = {entry.name for entry in run_dir.iterdir()}
    if entries and not entries.issubset(STAGING_FILES):
        raise FileExistsError(
            f"Run directory already contains training artifacts: {run_dir}. "
            "Use --resume true or choose another --name."
        )


def _resolve_dataset_root(data_yaml: Path, config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("path") or data_yaml.parent)).expanduser()
    return configured.resolve() if configured.is_absolute() else (data_yaml.parent / configured).resolve()


def _localize_split(value: Any, source_root: Path) -> Any:
    if isinstance(value, list):
        return [_localize_split(item, source_root) for item in value]
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return value
    try:
        return path.resolve().relative_to(source_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Dataset split {path} is outside configured dataset root {source_root}.") from exc


def copy_dataset_to_local(data_yaml: str | Path, local_root: str | Path) -> Path:
    """Copy the configured dataset root to Colab local disk and rewrite its YAML."""

    source_yaml = Path(data_yaml).expanduser().resolve()
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_root = _resolve_dataset_root(source_yaml, config)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {source_root}")

    destination = Path(local_root).expanduser().resolve()
    marker = destination / ".fapn_dataset_source.json"
    expected_marker = {"source_yaml": str(source_yaml), "source_root": str(source_root)}
    if destination == source_root:
        destination.mkdir(parents=True, exist_ok=True)
    elif destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"Local dataset target is not a directory: {destination}")
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != expected_marker:
            raise FileExistsError(
                f"Local dataset directory already exists but is not a matching FaPN copy: {destination}"
            )
    else:
        shutil.copytree(source_root, destination)
        write_json(marker, expected_marker)

    local_config = dict(config)
    local_config["path"] = str(destination)
    for split in ("train", "val", "test"):
        if split in local_config:
            local_config[split] = _localize_split(local_config[split], source_root)
    local_yaml = destination / "data_fapn_local.yaml"
    local_yaml.write_text(
        yaml.safe_dump(local_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return local_yaml


def _split_paths(value: Any, root: Path) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for item in values:
        path = Path(str(item)).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path.is_dir():
            images.extend(sorted(file for file in path.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    image = Path(line.strip())
                    images.append(image.resolve() if image.is_absolute() else (root / image).resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise FileNotFoundError(f"Dataset split path not found or unsupported: {path}")
    return images


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indices:
        return image.with_suffix(".txt")
    parts[image_indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def audit_dataset(data_yaml: str | Path) -> dict[str, Any]:
    """Count train/val/test images and matching YOLO label files."""

    data_yaml = Path(data_yaml).expanduser().resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = _resolve_dataset_root(data_yaml, config)
    splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        if split not in config or config[split] in (None, ""):
            splits[split] = {"present": False, "images": 0, "labels": 0, "missing_labels": 0}
            continue
        images = _split_paths(config[split], root)
        labels = [_label_path(image) for image in images]
        existing_labels = sum(label.is_file() for label in labels)
        splits[split] = {
            "present": True,
            "images": len(images),
            "labels": existing_labels,
            "missing_labels": len(images) - existing_labels,
            "counts_equal": len(images) == existing_labels,
        }
    if not splits["train"]["present"] or not splits["val"]["present"]:
        raise RuntimeError("Dataset YAML must provide both train and val splits.")
    if not splits["train"]["images"] or not splits["val"]["images"]:
        raise RuntimeError("Train and val splits must each contain at least one image.")
    return {"data_yaml": str(data_yaml), "root": str(root), "splits": splits}


def runtime_report(device: str) -> dict[str, Any]:
    """Check pinned Ultralytics, Torch/Torchvision DCNv2, and requested GPU."""

    import torch
    import torchvision
    from torchvision.ops import DeformConv2d

    version = require_ultralytics_version()
    wants_cuda = str(device).lower() not in {"cpu", "none"}
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU was requested but torch.cuda.is_available() is false.")
    op_device = torch.device("cuda:0" if wants_cuda else "cpu")
    probe = DeformConv2d(8, 8, 3, padding=1, bias=True).to(op_device)
    x = torch.randn(1, 8, 8, 8, device=op_device)
    offset = torch.zeros(1, 144, 8, 8, device=op_device)
    mask = torch.full((1, 72, 8, 8), 0.5, device=op_device)
    with torch.inference_mode():
        output = probe(x, offset, mask)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Torchvision modulated DeformConv2d runtime probe produced non-finite values.")
    return {
        "ultralytics": version,
        "required_ultralytics": EXPECTED_ULTRALYTICS_VERSION,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "requested_device": str(device),
        "deform_conv2d_probe": "passed",
    }


def resolved_default_training_policy() -> dict[str, Any]:
    """Record unchanged Ultralytics defaults for LR and augmentation."""

    from ultralytics.cfg import DEFAULT_CFG_DICT

    keys = (
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
        "close_mosaic",
    )
    return {key: DEFAULT_CFG_DICT[key] for key in keys}


def build_parser(variant: str) -> argparse.ArgumentParser:
    config = variant_config(variant)
    parser = argparse.ArgumentParser(description=f"Train {config['name']} with the formal shared protocol.")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--data", required=True, help="Source dataset YAML, typically on Google Drive.")
    parser.add_argument("--local-data-root", default="/content/datasets/ship_detection")
    parser.add_argument("--project", default="/content/drive/MyDrive/ship_detection/runs")
    parser.add_argument("--name", default=config["name"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--resume", type=parse_bool, default=False)
    return parser


def run_training(variant: str) -> None:
    """Prepare and execute one formal run in the current Python process."""

    args = build_parser(variant).parse_args()
    runtime = runtime_report(args.device)
    local_data_yaml = copy_dataset_to_local(args.data, args.local_data_root)
    dataset = audit_dataset(local_data_yaml)
    config = variant_config(variant)
    project = Path(args.project).expanduser().resolve()
    run_dir = project / args.name

    initialization_model = build_model(variant)
    structure = structure_report(initialization_model, variant)
    if not structure["all_checks_passed"]:
        raise RuntimeError(f"FaPN structure validation failed: {structure}")
    transfer = semantic_weight_transfer(initialization_model, args.weights, apply=not args.resume)
    stats = model_statistics(initialization_model, imgsz=args.imgsz)

    if args.resume:
        last_pt = validate_resume_checkpoint(run_dir / "weights" / "last.pt")
        from ultralytics import YOLO

        model = YOLO(str(last_pt))
        resume_source = str(last_pt)
    else:
        prepare_run_directory(run_dir)
        model = initialization_model
        resume_source = None

    metadata = {
        "variant": variant,
        "experiment_name": args.name,
        "model_yaml": str(Path(config["yaml"]).resolve()),
        "official_weights": str(args.weights),
        "source_data_yaml": str(Path(args.data).expanduser().resolve()),
        "local_data_yaml": str(local_data_yaml),
        "project": str(project),
        "run_dir": str(run_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "resume": args.resume,
        "resume_source": resume_source,
        "runtime": runtime,
        "dataset_audit": dataset,
        "ultralytics_default_lr_and_augmentation": resolved_default_training_policy(),
        "model_statistics": stats,
        "weight_transfer": transfer,
        "structure": structure,
        "git_commit": git_commit(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / METADATA_FILE, metadata)
    write_json(run_dir / TRANSFER_FILE, transfer)
    write_json(run_dir / DATASET_AUDIT_FILE, dataset)

    print(f"FaPN formal training request: {args.name}")
    print(f"  model: {metadata['model_yaml']}")
    print(f"  local data: {local_data_yaml}")
    for split, counts in dataset["splits"].items():
        print(f"  {split}: images={counts['images']} labels={counts['labels']} present={counts['present']}")
    print(f"  runtime: {runtime}")
    print(f"  parameters: {stats['parameters']}")
    print(f"  GFLOPs: {stats['gflops']:.6f}")
    print(
        "  inherited parameter elements: "
        f"{transfer['inherited_parameter_elements']}/{transfer['target_parameter_elements']} "
        f"({transfer['parameter_element_inheritance_ratio']:.4%})"
    )
    print(f"  run directory: {run_dir}")
    print(f"  resume: {args.resume}")

    if args.resume:
        result = model.train(resume=True)
    else:
        result = model.train(
            data=str(local_data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            optimizer=args.optimizer,
            project=str(project),
            name=args.name,
            exist_ok=True,
        )

    summary = {
        "experiment_name": args.name,
        "metrics": getattr(result, "results_dict", None) or {},
        "best": str(getattr(result, "best", "")),
        "last": str(getattr(result, "last", "")),
        "git_commit": git_commit(),
        "ultralytics_version": require_ultralytics_version(),
    }
    write_json(run_dir / "summary.json", summary)
