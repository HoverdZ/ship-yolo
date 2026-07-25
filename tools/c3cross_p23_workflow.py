"""C3Cross-P23 structure, hybrid initialization, AP75, and Colab training helpers.

The experiment deliberately preserves the ordinary 3x3 stride-2 downsampling
convolutions at backbone layers 1 and 3. Only the following C3k2 blocks at
layers 2 and 4 use C3k2CrossConv.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, Conv
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.c3k2_crossconv import C3k2CrossConv
from custom_modules.register import register_module_ablation_modules

EXPERIMENT_NAME = "yolo11n-c3cross-p23"
MODEL_YAML = ROOT / "experiments/yolo11n-c3cross-p23.yaml"
CROSS_PREFIXES = ("model.2.", "model.4.")
MAP_COLUMN = "metrics/mAP50-95(B)"
MAP50_COLUMN = "metrics/mAP50(B)"
PRECISION_COLUMN = "metrics/precision(B)"
RECALL_COLUMN = "metrics/recall(B)"


@dataclass(frozen=True)
class ScreeningThresholds:
    """Decision thresholds fixed before looking at the P2/P3-only result."""

    checkpoint_epoch: int = 15
    checkpoint_map50_95: float = 0.320
    checkpoint_recall: float = 0.700
    promotion_map50_95: float = 0.324
    promotion_recall: float = 0.705
    promotion_map50: float = 0.770


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _conv_signature(layer: nn.Module) -> dict[str, Any]:
    if not isinstance(layer, Conv):
        return {"type": type(layer).__name__, "kernel": None, "stride": None}
    return {
        "type": type(layer).__name__,
        "kernel": list(layer.conv.kernel_size),
        "stride": list(layer.conv.stride),
    }


def build_p23_model(*, nc: int | None = None) -> YOLO:
    """Build the registered P2/P3-only model, optionally for a dataset class count."""
    register_module_ablation_modules()
    wrapper = YOLO(str(MODEL_YAML), verbose=False)
    current_nc = int(wrapper.model.model[-1].nc)
    if nc is not None and current_nc != int(nc):
        wrapper.model = DetectionModel(
            cfg=str(MODEL_YAML),
            ch=3,
            nc=int(nc),
            verbose=False,
        )
        wrapper.ckpt = {}
        wrapper.task = "detect"
    return wrapper


def structure_report(model: YOLO) -> dict[str, Any]:
    """Audit exact replacement locations and protected downsampling convolutions."""
    layers = model.model.model
    cross_indices = [
        index
        for index, layer in enumerate(layers)
        if isinstance(layer, C3k2CrossConv)
    ]
    protected_downsamples = {
        str(index): _conv_signature(layers[index]) for index in (1, 3)
    }
    p4_p5_types = {
        "6": type(layers[6]).__name__,
        "8": type(layers[8]).__name__,
    }
    detect_from = list(layers[-1].f)
    downsample_passed = all(
        item["type"] == "Conv"
        and item["kernel"] == [3, 3]
        and item["stride"] == [2, 2]
        for item in protected_downsamples.values()
    )
    p4_p5_passed = all(
        isinstance(layers[index], C3k2)
        and not isinstance(layers[index], C3k2CrossConv)
        for index in (6, 8)
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "crossconv_indices": cross_indices,
        "protected_downsamples": protected_downsamples,
        "p4_p5_types": p4_p5_types,
        "detect_from": detect_from,
        "crossconv_scope_passed": cross_indices == [2, 4],
        "protected_downsamples_passed": downsample_passed,
        "p4_p5_standard_c3k2_passed": p4_p5_passed,
        "three_scale_detect_passed": detect_from == [16, 19, 22],
        "passed": (
            cross_indices == [2, 4]
            and downsample_passed
            and p4_p5_passed
            and detect_from == [16, 19, 22]
        ),
    }


def cpu_forward_backward_report(
    model: YOLO,
    *,
    imgsz: int = 64,
) -> dict[str, Any]:
    """Run a bounded CPU forward/backward smoke check without a dataset."""
    network = model.model.cpu().train()
    generator = torch.Generator(device="cpu").manual_seed(0)
    image = torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        generator=generator,
        requires_grad=True,
    )
    output = network(image)

    def collect_tensors(value: Any) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [
                tensor
                for item in value.values()
                for tensor in collect_tensors(item)
            ]
        if isinstance(value, (list, tuple)):
            return [
                tensor
                for item in value
                for tensor in collect_tensors(item)
            ]
        return []

    tensors = collect_tensors(output)
    if not tensors:
        raise RuntimeError(f"Model forward returned no tensors: {type(output).__name__}")
    proxy_loss = sum(tensor.float().square().mean() for tensor in tensors)
    proxy_loss.backward()
    gradients = [
        parameter.grad
        for parameter in network.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return {
        "input_shape": list(image.shape),
        "output_shapes": [list(tensor.shape) for tensor in tensors],
        "proxy_loss": float(proxy_loss.detach()),
        "input_gradient_finite": bool(
            image.grad is not None and torch.isfinite(image.grad).all()
        ),
        "parameter_gradients": len(gradients),
        "parameter_gradients_finite": all(
            torch.isfinite(gradient).all().item() for gradient in gradients
        ),
        "passed": bool(
            image.grad is not None
            and torch.isfinite(image.grad).all()
            and gradients
            and all(torch.isfinite(gradient).all().item() for gradient in gradients)
        ),
    }


def _load_source(weights: str | Path) -> tuple[YOLO, dict[str, torch.Tensor]]:
    register_module_ablation_modules()
    source = YOLO(str(weights), verbose=False)
    state = {
        key: value.detach().cpu().float()
        for key, value in source.model.state_dict().items()
    }
    return source, state


def hybrid_initialize(
    baseline_weights: str | Path,
    c3cross_weights: str | Path,
) -> tuple[YOLO, dict[str, Any]]:
    """Build a P2/P3 model from baseline weights plus trained shallow CrossConv.

    Baseline tensors populate every exact-name, exact-shape target first. The
    complete state of layers 2 and 4 is then overwritten from the full C3Cross
    checkpoint. Every target tensor must have one audited source.
    """
    baseline_weights = Path(baseline_weights)
    c3cross_weights = Path(c3cross_weights)
    for path in (baseline_weights, c3cross_weights):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    baseline, baseline_state = _load_source(baseline_weights)
    c3cross, c3cross_state = _load_source(c3cross_weights)
    baseline_nc = int(baseline.model.model[-1].nc)
    c3cross_nc = int(c3cross.model.model[-1].nc)
    if baseline_nc != c3cross_nc:
        raise ValueError(
            f"Checkpoint class counts differ: baseline={baseline_nc}, "
            f"c3cross={c3cross_nc}"
        )

    target = build_p23_model(nc=baseline_nc)
    target_state = target.model.state_dict()

    baseline_matched = {
        key: tensor
        for key, tensor in baseline_state.items()
        if key in target_state and tuple(tensor.shape) == tuple(target_state[key].shape)
    }
    target.model.load_state_dict(baseline_matched, strict=False)

    p23_target_keys = {
        key for key in target_state if key.startswith(CROSS_PREFIXES)
    }
    c3cross_overlay = {
        key: tensor
        for key, tensor in c3cross_state.items()
        if key in p23_target_keys
        and tuple(tensor.shape) == tuple(target_state[key].shape)
    }
    missing_p23 = sorted(p23_target_keys.difference(c3cross_overlay))
    if missing_p23:
        raise RuntimeError(
            "C3Cross checkpoint does not fully cover P2/P3 target tensors: "
            f"{missing_p23}"
        )
    target.model.load_state_dict(c3cross_overlay, strict=False)

    baseline_final_keys = set(baseline_matched).difference(c3cross_overlay)
    final_loaded_keys = baseline_final_keys.union(c3cross_overlay)
    missing_final = sorted(set(target_state).difference(final_loaded_keys))
    if missing_final:
        raise RuntimeError(
            "Hybrid initialization left target tensors without a source: "
            f"{missing_final}"
        )

    report = {
        "experiment": EXPERIMENT_NAME,
        "model_yaml": MODEL_YAML.relative_to(ROOT).as_posix(),
        "dataset_nc": baseline_nc,
        "baseline_weights": str(baseline_weights),
        "c3cross_weights": str(c3cross_weights),
        "protected_downsample_layers": [1, 3],
        "c3cross_overlay_prefixes": list(CROSS_PREFIXES),
        "source_tensors": {
            "baseline": len(baseline_state),
            "c3cross": len(c3cross_state),
        },
        "loaded_tensors": {
            "baseline_final": len(baseline_final_keys),
            "c3cross_p23_overlay": len(c3cross_overlay),
            "loaded_total": len(final_loaded_keys),
            "target_total": len(target_state),
        },
        "loaded_total_text": f"{len(final_loaded_keys)}/{len(target_state)}",
        "baseline_final_keys": sorted(baseline_final_keys),
        "c3cross_p23_keys": sorted(c3cross_overlay),
        "missing_target_keys": missing_final,
        "structure": structure_report(target),
        "passed": (
            len(final_loaded_keys) == len(target_state)
            and not missing_final
            and structure_report(target)["passed"]
        ),
    }
    return target, report


def save_hybrid_initialization(
    model: YOLO,
    report: dict[str, Any],
    checkpoint_path: str | Path,
    report_path: str | Path,
) -> tuple[Path, Path]:
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    report_output = _write_json(report_path, report)
    return checkpoint, report_output


def validate_checkpoint_metrics(
    label: str,
    data_yaml: str | Path,
    weights: str | Path,
    *,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 2,
    device: int | str = 0,
    work_dir: str | Path = "/content/ap75_audit",
) -> dict[str, Any]:
    """Validate one checkpoint and expose the exact Ultralytics AP75 value."""
    data_yaml = Path(data_yaml)
    weights = Path(weights)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    register_module_ablation_modules()
    model = YOLO(str(weights), verbose=False)
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        augment=False,
        plots=False,
        save_json=False,
        project=str(work_dir),
        name=label,
        exist_ok=True,
        verbose=True,
    )
    return {
        "experiment": label,
        "weights": str(weights),
        "split": "val",
        "imgsz": imgsz,
        "batch": batch,
        "augment": False,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "AP75": float(metrics.box.map75),
        "mAP50-95": float(metrics.box.map),
    }


def ap75_audit(
    data_yaml: str | Path,
    baseline_weights: str | Path,
    c3cross_weights: str | Path,
    output_csv: str | Path,
    *,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 2,
    device: int | str = 0,
    work_dir: str | Path = "/content/ap75_audit",
) -> pd.DataFrame:
    """Run identical validation for both best checkpoints and save exact AP75."""
    rows: list[dict[str, Any]] = []
    for label, weights in (
        ("yolo11n-baseline", Path(baseline_weights)),
        ("yolo11n-c3cross", Path(c3cross_weights)),
    ):
        rows.append(
            validate_checkpoint_metrics(
                label,
                data_yaml,
                weights,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                work_dir=work_dir,
            )
        )

    frame = pd.DataFrame(rows)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Saved AP75 audit: {output}")
    return frame


def _best_row(results_csv: str | Path) -> dict[str, float | int]:
    frame = pd.read_csv(results_csv)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in (
        "epoch",
        PRECISION_COLUMN,
        RECALL_COLUMN,
        MAP50_COLUMN,
        MAP_COLUMN,
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    row = frame.loc[frame[MAP_COLUMN].idxmax()]
    return {
        "epoch": int(row["epoch"]),
        "precision": float(row[PRECISION_COLUMN]),
        "recall": float(row[RECALL_COLUMN]),
        "mAP50": float(row[MAP50_COLUMN]),
        "mAP50-95": float(row[MAP_COLUMN]),
    }


def make_epoch15_screen_callback(
    thresholds: ScreeningThresholds = ScreeningThresholds(),
):
    """Stop once at epoch 15 when the fixed continuation gates are not met."""

    def screen(trainer) -> None:
        completed_epoch = trainer.epoch + 1
        if completed_epoch != thresholds.checkpoint_epoch:
            return
        best = _best_row(trainer.csv)
        passed = (
            best["mAP50-95"] >= thresholds.checkpoint_map50_95
            and best["recall"] >= thresholds.checkpoint_recall
        )
        payload = {
            "completed_epoch": completed_epoch,
            "best_through_checkpoint": best,
            "thresholds": asdict(thresholds),
            "continue_to_epoch_30": passed,
        }
        _write_json(Path(trainer.save_dir) / "epoch15_screen.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not passed:
            trainer.stop = True
            print("Epoch-15 gates failed; stopping the screening run.")

    return screen


def summarize_screening_run(
    run_dir: str | Path,
    thresholds: ScreeningThresholds = ScreeningThresholds(),
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Training results not found: {results_csv}")
    frame = pd.read_csv(results_csv)
    best = _best_row(results_csv)
    completed_epochs = int(pd.to_numeric(frame["epoch"]).max())
    promoted = (
        completed_epochs >= 30
        and best["mAP50-95"] >= thresholds.promotion_map50_95
        and best["recall"] >= thresholds.promotion_recall
        and best["mAP50"] >= thresholds.promotion_map50
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "completed_epochs": completed_epochs,
        "best": best,
        "thresholds": asdict(thresholds),
        "promoted_to_finetune": promoted,
        "decision": (
            "promote to one 20-epoch fine-tune"
            if promoted
            else "stop C3Cross; do not run 150 epochs"
        ),
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize_finetune_run(run_dir: str | Path) -> dict[str, Any]:
    """Summarize the one allowed fine-tune against its fixed acceptance gates."""
    run_dir = Path(run_dir)
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Fine-tune results not found: {results_csv}")
    frame = pd.read_csv(results_csv)
    best = _best_row(results_csv)
    completed_epochs = int(pd.to_numeric(frame["epoch"]).max())
    accepted = best["mAP50-95"] >= 0.324 and best["recall"] >= 0.705
    summary = {
        "experiment": run_dir.name,
        "completed_epochs": completed_epochs,
        "best": best,
        "acceptance_thresholds": {
            "mAP50-95": 0.324,
            "recall": 0.705,
        },
        "accepted": accepted,
        "decision": "retain tuned winner" if accepted else "retain untuned winner",
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_p23_screening(
    data_yaml: str | Path,
    baseline_weights: str | Path,
    c3cross_weights: str | Path,
    project: str | Path,
    *,
    init_checkpoint: str | Path = "/content/yolo11n-c3cross-p23-hybrid-init.pt",
    init_report: str | Path = "/content/yolo11n-c3cross-p23-hybrid-audit.json",
    device: int | str = 0,
) -> dict[str, Any]:
    """Run the fixed 30-epoch P2/P3-only screening protocol."""
    project = Path(project)
    run_dir = project / EXPERIMENT_NAME
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists; use a new experiment name: {run_dir}"
        )

    model, report = hybrid_initialize(baseline_weights, c3cross_weights)
    save_hybrid_initialization(model, report, init_checkpoint, init_report)
    print(
        "Hybrid Loaded/Total tensors:",
        report["loaded_tensors"]["loaded_total"],
        "/",
        report["loaded_tensors"]["target_total"],
    )
    print(
        "  baseline final:",
        report["loaded_tensors"]["baseline_final"],
        "| C3Cross P2/P3:",
        report["loaded_tensors"]["c3cross_p23_overlay"],
    )

    def save_initialization_audit(trainer) -> None:
        destination = Path(trainer.save_dir) / "hybrid_initialization_audit.json"
        _write_json(destination, report)

    model.add_callback("on_pretrain_routine_start", save_initialization_audit)
    model.add_callback("on_fit_epoch_end", make_epoch15_screen_callback())
    model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=8,
        workers=2,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.1,
        warmup_epochs=1.0,
        mosaic=0.5,
        close_mosaic=5,
        scale=0.4,
        translate=0.1,
        patience=10,
        seed=0,
        deterministic=True,
        amp=True,
        pretrained=False,
        cache=False,
        val=True,
        save=True,
        plots=True,
        project=str(project),
        name=EXPERIMENT_NAME,
        exist_ok=False,
        device=device,
        verbose=True,
    )
    return summarize_screening_run(run_dir)


def run_winner_finetune(
    data_yaml: str | Path,
    winner_best_pt: str | Path,
    project: str | Path,
    *,
    name: str = "yolo11n-c3cross-p23-finetune",
    device: int | str = 0,
) -> dict[str, Any]:
    """Run the single allowed 20-epoch low-LR fine-tune after promotion."""
    winner_best_pt = Path(winner_best_pt)
    if not winner_best_pt.is_file():
        raise FileNotFoundError(f"Winner checkpoint not found: {winner_best_pt}")
    run_dir = Path(project) / name
    if run_dir.exists():
        raise FileExistsError(
            f"Fine-tune directory already exists; do not overwrite it: {run_dir}"
        )
    register_module_ablation_modules()
    model = YOLO(str(winner_best_pt), verbose=False)
    model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=640,
        batch=8,
        workers=2,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.1,
        warmup_epochs=1.0,
        mosaic=0.2,
        close_mosaic=5,
        scale=0.30,
        translate=0.05,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=8,
        seed=0,
        deterministic=True,
        amp=True,
        pretrained=False,
        cache=False,
        val=True,
        save=True,
        plots=True,
        project=str(project),
        name=name,
        exist_ok=False,
        device=device,
        verbose=True,
    )
    return summarize_finetune_run(run_dir)


def structure_check(output: str | Path | None = None, *, imgsz: int = 64) -> dict[str, Any]:
    model = build_p23_model()
    payload = {
        "structure": structure_report(model),
        "cpu_forward_backward": cpu_forward_backward_report(model, imgsz=imgsz),
    }
    payload["passed"] = bool(
        payload["structure"]["passed"]
        and payload["cpu_forward_backward"]["passed"]
    )
    if output is not None:
        _write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("structure",),
        nargs="?",
        default="structure",
    )
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "structure":
        report = structure_check(args.output, imgsz=args.imgsz)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit("C3Cross-P23 structure check failed.")


if __name__ == "__main__":
    main()
