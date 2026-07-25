"""Independent training entrypoint for cumulative YOLO11n experiments.

Every experiment starts from the same official ``yolo11n.pt`` source. The
entrypoint never chains a previous experiment's best.pt into the next model.
It prepares and audits initialization but does not run unless invoked.
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

from tools.cumulative_models_utils import (
    EXPERIMENTS,
    build_model,
    read_dataset_nc,
    transfer_pretrained_weights,
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
            "Choose a new --name; cumulative experiments must start independently."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENTS),
        required=True,
    )
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
    parser.add_argument(
        "--cache",
        choices=("ram", "disk", "none"),
        default="disk",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


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
    """Train one experiment in the current Python process.

    This callable is the Colab entrypoint. Keeping training in the notebook
    kernel preserves Ultralytics' live per-epoch logging and progress bars.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment!r}; "
            f"choose one of {tuple(EXPERIMENTS)}."
        )
    if cache not in {"ram", "disk", "none", False, None}:
        raise ValueError(
            "cache must be 'ram', 'disk', 'none', False, or None; "
            f"got {cache!r}."
        )
    resolved_cache = None if cache in {"none", False, None} else cache
    args = argparse.Namespace(
        experiment=experiment,
        data=str(data),
        weights=str(weights),
        project=str(project),
        name=name,
        device=device,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        cache=resolved_cache,
        deterministic=bool(deterministic),
    )
    data_yaml = Path(args.data).expanduser().resolve()
    dataset_nc = read_dataset_nc(data_yaml)
    project = Path(args.project).expanduser().resolve()
    name = args.name or args.experiment
    run_dir = project / name
    prepare_run_directory(run_dir)

    model = build_model(
        args.experiment,
        nc=dataset_nc,
        verbose=True,
    )
    transfer = transfer_pretrained_weights(
        model,
        args.weights,
        apply=True,
    )
    if not transfer["passed"]:
        raise RuntimeError(
            "Official weight inheritance verification failed: "
            f"{transfer['verification_failures']}"
        )

    inheritance_path = write_json(
        run_dir / "inheritance_report.json",
        transfer,
    )
    init_checkpoint = run_dir / "initialization.pt"
    model.save(init_checkpoint)

    # Reloading makes the checkpoint explicit to Ultralytics 8.4.92. Training
    # must keep pretrained=True; pretrained=False would discard this state
    # while the Trainer rebuilds its model.
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import unwrap_model

    train_model = YOLO(str(init_checkpoint), verbose=False)
    if not train_model.ckpt:
        raise RuntimeError("Initialization checkpoint reload produced no ckpt.")
    expected_state = {
        key: value.detach().cpu().clone()
        for key, value in train_model.model.state_dict().items()
    }

    resolved = {
        "experiment": args.experiment,
        "model_yaml": str(EXPERIMENTS[args.experiment]["yaml"]),
        "official_weights": str(args.weights),
        "data": str(data_yaml),
        "dataset_nc": dataset_nc,
        "project": str(project),
        "name": name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": 0,
        "optimizer": "auto",
        "cache": args.cache,
        "deterministic": args.deterministic,
        "inheritance_report": str(inheritance_path),
    }
    write_json(run_dir / "resolved_args.json", resolved)

    verification = {"executed": False, "passed": False}

    def verify_trainer_state(trainer) -> None:
        actual = {
            key: value.detach().cpu()
            for key, value in unwrap_model(
                trainer.model
            ).state_dict().items()
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
                "passed": not missing
                and not unexpected
                and not mismatched,
                "loaded_total": (
                    f"{len(actual) - len(mismatched)}/{len(expected_state)}"
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
                "Trainer discarded or changed the audited initialization."
            )

    train_model.add_callback(
        "on_pretrain_routine_end",
        verify_trainer_state,
    )
    train_model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project),
        name=name,
        exist_ok=True,
        patience=150,
        cache=args.cache,
        optimizer="auto",
        seed=0,
        # DySample uses CUDA grid_sample backward, for which PyTorch has no
        # deterministic implementation. Setting this False prevents PyTorch's
        # warning stack from corrupting the official tqdm epoch display.
        deterministic=args.deterministic,
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
        raise RuntimeError("Trainer weight verification did not pass.")


def main() -> None:
    """CLI wrapper around the same in-process training function."""

    train_experiment(**vars(parse_args()))


if __name__ == "__main__":
    main()
