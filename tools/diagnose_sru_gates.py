#!/usr/bin/env python3
"""Collect exact SRU sigmoid/gate statistics for ship and background validation images."""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import torch
import yaml


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def add_repo_root_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def register_project_modules() -> None:
    module = importlib.import_module("custom_modules.register")

    for function_name in ("register_custom_modules", "register_modules", "register"):
        function = getattr(module, function_name, None)
        if not callable(function):
            continue

        signature = inspect.signature(function)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if not required:
            function()
            return

    print("Registration module imported; no zero-argument registration function was required.")


def chunks(items: list[Path], size: int) -> Iterator[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def resolve_dataset_root(yaml_path: Path, config: dict[str, Any]) -> Path:
    root_value = config.get("path")
    if root_value is None:
        return yaml_path.parent.resolve()

    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def resolve_entry_path(raw: str, dataset_root: Path, yaml_path: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidate_from_root = (dataset_root / path).resolve()
    if candidate_from_root.exists():
        return candidate_from_root

    return (yaml_path.parent / path).resolve()


def read_image_list_file(list_path: Path, dataset_root: Path) -> list[Path]:
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        path = Path(line).expanduser()
        if not path.is_absolute():
            candidates = [
                (dataset_root / path).resolve(),
                (list_path.parent / path).resolve(),
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

        if path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path.resolve())

    return images


def collect_images_from_entry(
    entry: str | list[str],
    dataset_root: Path,
    yaml_path: Path,
) -> list[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []

    for raw_entry in entries:
        path = resolve_entry_path(str(raw_entry), dataset_root, yaml_path)

        if path.is_dir():
            images.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            images.extend(read_image_list_file(path, dataset_root))
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise FileNotFoundError(f"Dataset split entry does not exist or is unsupported: {path}")

    return sorted(set(images))


def infer_label_path(image_path: Path, dataset_root: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]

    if image_indices:
        index = image_indices[-1]
        label_parts = parts.copy()
        label_parts[index] = "labels"
        return Path(*label_parts).with_suffix(".txt")

    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        relative = Path(image_path.name)

    candidates = [
        dataset_root / "labels" / relative.with_suffix(".txt"),
        image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def has_nonempty_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False

    try:
        return any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return label_path.stat().st_size > 0


def split_ship_and_background(
    images: list[Path],
    dataset_root: Path,
) -> tuple[list[Path], list[Path]]:
    ship_images: list[Path] = []
    background_images: list[Path] = []

    for image_path in images:
        label_path = infer_label_path(image_path, dataset_root)
        if has_nonempty_label(label_path):
            ship_images.append(image_path)
        else:
            background_images.append(image_path)

    return ship_images, background_images


def find_sru_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__.lower() == "sru"
    ]
    if not modules:
        raise RuntimeError("No SRU modules were found in the loaded model.")
    return modules


def gate_threshold(module: torch.nn.Module) -> float:
    for attribute_name in ("gate_threshold", "gate_treshold", "threshold"):
        value = getattr(module, attribute_name, None)
        if value is not None:
            return float(value)
    return 0.5


def sigmoid_children(module: torch.nn.Module) -> list[torch.nn.Module]:
    return [
        child
        for child in module.modules()
        if child is not module and isinstance(child, torch.nn.Sigmoid)
    ]


@dataclass
class RunningGateStats:
    element_count: int = 0
    high_count: int = 0
    near_001_count: int = 0
    near_005_count: int = 0
    weight_sum: float = 0.0
    weight_sq_sum: float = 0.0
    weight_min: float = math.inf
    weight_max: float = -math.inf
    sample_ratios: list[float] = field(default_factory=list)

    def update(self, weights: torch.Tensor, threshold: float) -> None:
        values = weights.detach().float()
        if values.ndim < 2:
            raise RuntimeError(f"Unexpected SRU gate tensor shape: {tuple(values.shape)}")

        total = values.numel()
        high = values >= threshold

        self.element_count += total
        self.high_count += int(high.sum().item())
        self.near_001_count += int((values.sub(threshold).abs() <= 0.01).sum().item())
        self.near_005_count += int((values.sub(threshold).abs() <= 0.05).sum().item())
        self.weight_sum += float(values.sum().item())
        self.weight_sq_sum += float(values.square().sum().item())
        self.weight_min = min(self.weight_min, float(values.min().item()))
        self.weight_max = max(self.weight_max, float(values.max().item()))

        batch_size = values.shape[0]
        per_sample = high.reshape(batch_size, -1).float().mean(dim=1)
        self.sample_ratios.extend(float(value) for value in per_sample.cpu().tolist())

    def as_dict(self) -> dict[str, float | int]:
        if self.element_count == 0:
            return {
                "element_count": 0,
                "sample_count": 0,
                "gate_ratio": float("nan"),
                "weight_mean": float("nan"),
                "weight_std": float("nan"),
                "weight_min": float("nan"),
                "weight_max": float("nan"),
                "near_threshold_0.01": float("nan"),
                "near_threshold_0.05": float("nan"),
                "sample_gate_ratio_mean": float("nan"),
                "sample_gate_ratio_std": float("nan"),
                "sample_gate_ratio_min": float("nan"),
                "sample_gate_ratio_max": float("nan"),
            }

        mean = self.weight_sum / self.element_count
        variance = max(self.weight_sq_sum / self.element_count - mean * mean, 0.0)
        ratios = self.sample_ratios

        return {
            "element_count": self.element_count,
            "sample_count": len(ratios),
            "gate_ratio": self.high_count / self.element_count,
            "weight_mean": mean,
            "weight_std": math.sqrt(variance),
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "near_threshold_0.01": self.near_001_count / self.element_count,
            "near_threshold_0.05": self.near_005_count / self.element_count,
            "sample_gate_ratio_mean": float(np.mean(ratios)),
            "sample_gate_ratio_std": float(np.std(ratios)),
            "sample_gate_ratio_min": min(ratios),
            "sample_gate_ratio_max": max(ratios),
        }


class GateCollector:
    def __init__(self) -> None:
        self.current_group: str | None = None
        self.stats: dict[tuple[str, str], RunningGateStats] = defaultdict(RunningGateStats)
        self.handles: list[Any] = []
        self.capture_mode: dict[str, str] = {}

    def attach(self, model: torch.nn.Module) -> list[str]:
        layer_names: list[str] = []

        for layer_name, sru in find_sru_modules(model):
            layer_names.append(layer_name)
            threshold = gate_threshold(sru)
            sigmoid_modules = sigmoid_children(sru)

            if sigmoid_modules:
                # Capture the exact tensor produced by the SRU's own Sigmoid module.
                sigmoid_module = sigmoid_modules[0]

                def make_exact_hook(name: str, gate_value: float):
                    def hook(
                        _module: torch.nn.Module,
                        _inputs: tuple[Any, ...],
                        output: torch.Tensor,
                    ) -> None:
                        if self.current_group is None:
                            return
                        self.stats[(self.current_group, name)].update(output, gate_value)

                    return hook

                self.handles.append(
                    sigmoid_module.register_forward_hook(make_exact_hook(layer_name, threshold))
                )
                self.capture_mode[layer_name] = "exact_sigmoid_hook"
            else:
                # Fallback for an implementation that uses torch.sigmoid functionally.
                def make_fallback_hook(name: str, parent: torch.nn.Module, gate_value: float):
                    def pre_hook(
                        _module: torch.nn.Module,
                        inputs: tuple[Any, ...],
                    ) -> None:
                        if self.current_group is None:
                            return
                        if not inputs or not torch.is_tensor(inputs[0]):
                            return

                        x = inputs[0]
                        normalizer = getattr(parent, "gn", None)
                        if normalizer is None:
                            raise RuntimeError(
                                f"SRU {name} has no Sigmoid child and no 'gn' attribute for fallback."
                            )

                        normalized = normalizer(x)
                        gamma = getattr(normalizer, "gamma", None)
                        if gamma is None:
                            gamma = getattr(normalizer, "weight", None)
                        if gamma is None:
                            gamma = torch.ones(
                                x.shape[1],
                                device=x.device,
                                dtype=x.dtype,
                            )

                        gamma = gamma.reshape(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
                        eps = float(getattr(normalizer, "eps", 1e-5))
                        denominator = gamma.sum()
                        denominator = denominator + denominator.new_tensor(eps)
                        weights = torch.sigmoid(normalized * (gamma / denominator))
                        self.stats[(self.current_group, name)].update(weights, gate_value)

                    return pre_hook

                self.handles.append(
                    sru.register_forward_pre_hook(make_fallback_hook(layer_name, sru, threshold))
                )
                self.capture_mode[layer_name] = "formula_fallback"

        return layer_names

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def preprocess_batch(paths: list[Path], imgsz: int, device: torch.device) -> torch.Tensor:
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, stride=32)
    tensors: list[torch.Tensor] = []

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")

        image = letterbox(image=image)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).float().div_(255.0)
        tensors.append(tensor)

    return torch.stack(tensors, dim=0).to(device=device, non_blocking=True)


def run_group(
    raw_model: torch.nn.Module,
    collector: GateCollector,
    group_name: str,
    paths: list[Path],
    imgsz: int,
    batch_size: int,
    device: torch.device,
) -> None:
    collector.current_group = group_name

    with torch.inference_mode():
        for index, batch_paths in enumerate(chunks(paths, batch_size), start=1):
            batch = preprocess_batch(batch_paths, imgsz, device)
            _ = raw_model(batch)

            if index % 10 == 0 or index == math.ceil(len(paths) / batch_size):
                processed = min(index * batch_size, len(paths))
                print(f"{group_name}: {processed}/{len(paths)} images")

    collector.current_group = None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact SRU sigmoid/gate behavior separately on ship and background images."
        )
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="One or more checkpoints, e.g. best.pt last.pt.",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples-per-group", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="diagnostics/sru_gates",
        help="Output directory, relative to repository root unless absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = add_repo_root_to_path()
    register_project_modules()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from ultralytics import YOLO

    yaml_path = Path(args.data).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML is not a dictionary.")
    if args.split not in config:
        raise KeyError(f"Dataset YAML has no split named '{args.split}'.")

    dataset_root = resolve_dataset_root(yaml_path, config)
    images = collect_images_from_entry(config[args.split], dataset_root, yaml_path)
    ship_images, background_images = split_ship_and_background(images, dataset_root)

    if not ship_images:
        raise RuntimeError("No labeled ship images were found.")
    if not background_images:
        raise RuntimeError("No background images were found.")

    random.shuffle(ship_images)
    random.shuffle(background_images)
    ship_images = ship_images[: min(args.samples_per_group, len(ship_images))]
    background_images = background_images[
        : min(args.samples_per_group, len(background_images))
    ]

    print(f"Dataset root: {dataset_root}")
    print(f"Total {args.split} images: {len(images)}")
    print(f"Available ship images: {len(split_ship_and_background(images, dataset_root)[0])}")
    print(f"Available background images: {len(split_ship_and_background(images, dataset_root)[1])}")
    print(f"Selected ship images: {len(ship_images)}")
    print(f"Selected background images: {len(background_images)}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        device = torch.device(f"cuda:{args.device}")

    all_rows: list[dict[str, Any]] = []
    checkpoint_reports: list[dict[str, Any]] = []

    for weight_value in args.weights:
        weight_path = Path(weight_value).expanduser().resolve()
        if not weight_path.exists():
            raise FileNotFoundError(weight_path)

        print(f"\n{'=' * 80}")
        print(f"Checkpoint: {weight_path}")
        print(f"{'=' * 80}")

        yolo = YOLO(str(weight_path))
        raw_model = yolo.model.to(device).float().eval()

        collector = GateCollector()
        layer_names = collector.attach(raw_model)
        print(f"Found {len(layer_names)} SRU modules.")
        if len(layer_names) != 12:
            print(
                "WARNING: the experiment report expected 12 SRUs, "
                f"but the loaded checkpoint contains {len(layer_names)}."
            )

        try:
            run_group(
                raw_model,
                collector,
                "ship",
                ship_images,
                args.imgsz,
                args.batch,
                device,
            )
            run_group(
                raw_model,
                collector,
                "background",
                background_images,
                args.imgsz,
                args.batch,
                device,
            )
        finally:
            collector.close()

        checkpoint_name = weight_path.stem
        rows_for_checkpoint: list[dict[str, Any]] = []

        for layer_name in layer_names:
            threshold_value = gate_threshold(dict(raw_model.named_modules())[layer_name])
            for group_name in ("ship", "background"):
                result = collector.stats[(group_name, layer_name)].as_dict()
                row = {
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": str(weight_path),
                    "layer": layer_name,
                    "capture_mode": collector.capture_mode[layer_name],
                    "group": group_name,
                    "gate_threshold": threshold_value,
                    **result,
                }
                rows_for_checkpoint.append(row)
                all_rows.append(row)

        write_csv(output_dir / f"{checkpoint_name}_sru_gate_stats.csv", rows_for_checkpoint)

        by_layer_group = {
            (row["layer"], row["group"]): row for row in rows_for_checkpoint
        }
        deltas = []
        for layer_name in layer_names:
            ship = by_layer_group[(layer_name, "ship")]
            background = by_layer_group[(layer_name, "background")]
            delta = float(ship["gate_ratio"]) - float(background["gate_ratio"])
            deltas.append(
                {
                    "layer": layer_name,
                    "ship_gate_ratio": ship["gate_ratio"],
                    "background_gate_ratio": background["gate_ratio"],
                    "ship_minus_background": delta,
                    "ship_near_0.01": ship["near_threshold_0.01"],
                    "background_near_0.01": background["near_threshold_0.01"],
                }
            )

        checkpoint_report = {
            "checkpoint": str(weight_path),
            "sru_count": len(layer_names),
            "capture_modes": collector.capture_mode,
            "selected_images": {
                "ship": [str(path) for path in ship_images],
                "background": [str(path) for path in background_images],
            },
            "rows": rows_for_checkpoint,
            "ship_background_deltas": deltas,
        }
        checkpoint_reports.append(checkpoint_report)

        print("\nLayer gate summary")
        print(
            f"{'layer':45s} {'ship':>9s} {'background':>12s} "
            f"{'delta':>9s} {'near±.01 ship':>15s} {'near±.01 bg':>13s}"
        )
        for item in deltas:
            print(
                f"{item['layer'][:45]:45s} "
                f"{item['ship_gate_ratio']:9.4f} "
                f"{item['background_gate_ratio']:12.4f} "
                f"{item['ship_minus_background']:9.4f} "
                f"{item['ship_near_0.01']:15.4f} "
                f"{item['background_near_0.01']:13.4f}"
            )

        del raw_model
        del yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output_dir / "all_checkpoints_sru_gate_stats.csv", all_rows)
    report_path = output_dir / "sru_gate_report.json"
    report_path.write_text(
        json.dumps(
            {
                "data": str(yaml_path),
                "split": args.split,
                "settings": {
                    "samples_per_group": args.samples_per_group,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "device": args.device,
                    "seed": args.seed,
                },
                "checkpoints": checkpoint_reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nReports saved to: {output_dir}")
    print(f"Combined CSV: {output_dir / 'all_checkpoints_sru_gate_stats.csv'}")
    print(f"JSON report: {report_path}")
    print(
        "\nInterpretation priorities:\n"
        "1. Gate ratios close to 0 or 1 indicate saturation.\n"
        "2. A large fraction within ±0.01 of threshold 0.5 means tiny parameter changes "
        "can flip many hard gates.\n"
        "3. Very small ship-background deltas mean SRU does not distinguish target and background.\n"
        "4. Large sample-level gate-ratio standard deviation means behavior changes strongly by image."
    )


if __name__ == "__main__":
    main()
