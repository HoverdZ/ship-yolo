"""Official in-process Ultralytics training entrypoint for CA-SCAM.

Colab imports and calls ``train_ca_scam`` directly. Never wrap this training
function in a subprocess: the official per-epoch progress remains visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.calibrated_scam_utils import (
    MODEL_YAML,
    build_model,
    initialize_from_official,
    structure_report,
    write_json,
)
from tools.cumulative_models_utils import read_dataset_nc

ALLOWED_STAGING_FILES = {
    "initialization.pt",
    "inheritance_report.json",
    "resolved_args.json",
    "structure_report.json",
}


def prepare_run_directory(run_dir: Path) -> None:
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"Run path is not a directory: {run_dir}")
    entries = {item.name for item in run_dir.iterdir()}
    if entries and not entries.issubset(ALLOWED_STAGING_FILES):
        raise FileExistsError(
            f"Run directory contains training artifacts: {run_dir}. "
            "Choose a new name; this experiment must start independently."
        )


def train_ca_scam(
    *,
    data: str | Path,
    weights: str | Path = "yolo11n.pt",
    project: str | Path = "/content/drive/MyDrive/ship_detection/runs",
    name: str = "yolo11n_incdw_dysample_pls_ca_scam_vgup_640",
    device: str = "0",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 2,
    cache: str | bool | None = "disk",
    deterministic: bool = False,
) -> None:
    """Train CA-SCAM through Ultralytics' official API in the current kernel."""

    if cache not in {"ram", "disk", "none", False, None}:
        raise ValueError(f"Unsupported cache mode: {cache!r}")
    resolved_cache = None if cache in {"none", False, None} else cache
    data_yaml = Path(data).expanduser().resolve()
    dataset_nc = read_dataset_nc(data_yaml)
    project_path = Path(project).expanduser().resolve()
    run_dir = project_path / name
    prepare_run_directory(run_dir)

    model = build_model(nc=dataset_nc, verbose=True)
    structure = structure_report(model)
    if not structure["passed"]:
        raise RuntimeError(f"CA-SCAM topology audit failed: {structure}")
    inheritance = initialize_from_official(
        model,
        weights=weights,
        apply=True,
    )
    if not inheritance["passed"]:
        raise RuntimeError(f"CA-SCAM inheritance audit failed: {inheritance}")
    write_json(run_dir / "structure_report.json", structure)
    write_json(run_dir / "inheritance_report.json", inheritance)

    initialization_path = run_dir / "initialization.pt"
    model.save(initialization_path)

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import unwrap_model

    training_model = YOLO(str(initialization_path), verbose=False)
    expected_state = {
        key: value.detach().cpu().clone()
        for key, value in training_model.model.state_dict().items()
    }
    resolved = {
        "experiment": "incdw_dysample_pls_ca_scam_vgup",
        "model_yaml": str(MODEL_YAML),
        "official_weights": str(weights),
        "uses_successful_best_pt": False,
        "data": str(data_yaml),
        "dataset_nc": dataset_nc,
        "project": str(project_path),
        "name": name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "cache": resolved_cache,
        "deterministic": deterministic,
        "optimizer": "auto",
        "seed": 0,
    }
    write_json(run_dir / "resolved_args.json", resolved)

    verification = {"executed": False, "passed": False}

    def verify_trainer_state(trainer) -> None:
        actual = {
            key: value.detach().cpu()
            for key, value in unwrap_model(trainer.model).state_dict().items()
        }
        missing = sorted(set(expected_state) - set(actual))
        unexpected = sorted(set(actual) - set(expected_state))
        mismatched = [
            key
            for key, value in expected_state.items()
            if key in actual and not torch.equal(value, actual[key])
        ]
        verification.update(
            {
                "executed": True,
                "passed": not missing and not unexpected and not mismatched,
                "loaded_total": (
                    f"{len(expected_state) - len(mismatched)}/"
                    f"{len(expected_state)}"
                ),
                "missing": missing,
                "unexpected": unexpected,
                "mismatched": mismatched,
            }
        )
        write_json(
            Path(trainer.save_dir) / "trainer_weight_verification.json",
            verification,
        )
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        if not verification["passed"]:
            raise RuntimeError(
                "Ultralytics Trainer changed the audited initialization."
            )

    training_model.add_callback(
        "on_pretrain_routine_end",
        verify_trainer_state,
    )
    # Official API, current Python kernel, live Ultralytics progress output.
    training_model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(project_path),
        name=name,
        exist_ok=True,
        patience=150,
        cache=resolved_cache,
        optimizer="auto",
        seed=0,
        deterministic=deterministic,
        pretrained=True,
        resume=False,
        amp=True,
        compile=False,
        close_mosaic=10,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        val=True,
        save=True,
        plots=True,
        verbose=True,
    )
    if not verification["executed"] or not verification["passed"]:
        raise RuntimeError("Trainer initialization verification did not pass.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument(
        "--project",
        default="/content/drive/MyDrive/ship_detection/runs",
    )
    parser.add_argument(
        "--name",
        default="yolo11n_incdw_dysample_pls_ca_scam_vgup_640",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache", choices=("ram", "disk", "none"), default="disk")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train_ca_scam(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
