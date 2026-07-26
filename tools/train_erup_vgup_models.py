"""Direct Ultralytics training entrypoint for the four ERUP/VGUP models.

Formal training is intentionally not started by this file. Colab imports
``train_experiment`` and calls it in the current kernel so the official
Ultralytics progress table remains live and visible.
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

from tools.cumulative_models_utils import read_dataset_nc
from tools.erup_vgup_utils import (
    EXPERIMENTS,
    build_model,
    initialize_from_official,
    write_json,
)

ALLOWED_STAGING_FILES = {
    "initialization.pt",
    "inheritance_report.json",
    "resolved_args.json",
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
            "Choose a new name; every comparison must start independently."
        )


def train_experiment(
    *,
    experiment: str,
    data: str | Path,
    weights: str | Path = "yolo11n.pt",
    project: str | Path = "/content/drive/MyDrive/ship_detection/runs",
    name: str | None = None,
    device: str = "0",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 2,
    cache: str | bool | None = "disk",
    deterministic: bool = False,
) -> None:
    """Train one model through the official API in the current process."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment!r}")
    if cache not in {"ram", "disk", "none", False, None}:
        raise ValueError(f"Unsupported cache mode: {cache!r}")
    resolved_cache = None if cache in {"none", False, None} else cache
    data_yaml = Path(data).expanduser().resolve()
    dataset_nc = read_dataset_nc(data_yaml)
    project_path = Path(project).expanduser().resolve()
    run_name = name or experiment
    run_dir = project_path / run_name
    prepare_run_directory(run_dir)

    model = build_model(experiment, nc=dataset_nc, verbose=True)
    inheritance = initialize_from_official(
        model,
        experiment,
        weights=weights,
        apply=True,
    )
    if not inheritance["passed"]:
        raise RuntimeError(
            "Detector inheritance failed: "
            f"{inheritance['verification_failures']}"
        )
    inheritance_path = write_json(
        run_dir / "inheritance_report.json",
        inheritance,
    )
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
        "experiment": experiment,
        "model_yaml": str(EXPERIMENTS[experiment]["yaml"]),
        "official_weights": str(weights),
        "data": str(data_yaml),
        "dataset_nc": dataset_nc,
        "project": str(project_path),
        "name": run_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "cache": resolved_cache,
        "deterministic": deterministic,
        "optimizer": "auto",
        "seed": 0,
        "inheritance_report": str(inheritance_path),
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
    # This is the official in-process API. Do not wrap it with subprocess.
    training_model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(project_path),
        name=run_name,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument(
        "--project",
        default="/content/drive/MyDrive/ship_detection/runs",
    )
    parser.add_argument("--name")
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
    return parser.parse_args()


def main() -> None:
    train_experiment(**vars(parse_args()))


if __name__ == "__main__":
    main()

