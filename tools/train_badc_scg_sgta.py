"""Direct-process staged training for BADC, SCG, SGTA, and the full candidate."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.badc_scg_sgta_utils import (
    VARIANTS,
    git_commit,
    require_ultralytics_version,
    save_initialized_model,
    variant_config,
    write_json,
)


METADATA_FILE = "experiment_metadata.json"
STAGE_CHECKPOINT = "stage80_resume.pt"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class TrainingRequest:
    variant: str
    data: str
    project: str
    weights: str = "yolo11n.pt"
    name: str | None = None
    total_epochs: int = 150
    stage_epochs: int = 80
    imgsz: int = 640
    batch: int = 8
    workers: int = 2
    device: str = "0"
    seed: int = 0
    optimizer: str = "auto"
    resume: bool = False
    resume_checkpoint: str | None = None


def validate_data_yaml(path: str | Path) -> Path:
    data = Path(path).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}; no fallback is allowed.")
    return data


def validate_resume_checkpoint(path: str | Path, expected_total_epochs: int) -> Path:
    import torch

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint type: {type(payload).__name__}")
    missing = [key for key in ("epoch", "optimizer", "train_args") if payload.get(key) is None]
    if missing:
        raise RuntimeError(f"Checkpoint is not resumable; missing {missing}: {checkpoint}")
    configured_epochs = int(payload["train_args"].get("epochs", -1))
    if configured_epochs != expected_total_epochs:
        raise RuntimeError(
            f"Checkpoint total epochs={configured_epochs}, expected {expected_total_epochs}. "
            "Do not restart the scheduler with a different horizon."
        )
    return checkpoint


def prepare_new_run_directory(run_dir: Path) -> None:
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"Run path is not a directory: {run_dir}")
    entries = {entry.name for entry in run_dir.iterdir()}
    if entries and not entries.issubset({METADATA_FILE}):
        raise FileExistsError(
            f"Run directory already contains artifacts: {run_dir}. "
            "Use the resume cell or choose another run name."
        )


def _sgta_trainer_class():
    from ultralytics.models.yolo.detect import DetectionTrainer
    from custom_modules.sgta import SGTADetectionModel

    class SGTATrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                SGTADetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                )
            )
            if weights:
                model.load(weights)
            return model

    return SGTATrainer


def _attach_stage_callbacks(model, run_dir: Path, stage_epochs: int) -> None:
    """Stop after the screening stage and preserve a resumable raw checkpoint."""

    stage_path = run_dir / "weights" / STAGE_CHECKPOINT

    def stop_at_stage(trainer):
        if trainer.epoch + 1 == stage_epochs:
            print(
                f"\nScreening stage reached epoch {stage_epochs}; "
                "saving resumable optimizer/scheduler/EMA state."
            )
            trainer.stop = True

    def preserve_stage_checkpoint(trainer):
        if trainer.epoch + 1 == stage_epochs:
            if not trainer.last.is_file():
                raise FileNotFoundError(
                    f"Ultralytics did not create the raw checkpoint: {trainer.last}"
                )
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trainer.last, stage_path)
            print(f"Preserved resumable stage checkpoint: {stage_path}")

    model.add_callback("on_train_epoch_end", stop_at_stage)
    model.add_callback("on_model_save", preserve_stage_checkpoint)


def run_training(request: TrainingRequest):
    """Run training synchronously in the notebook kernel."""

    from custom_modules.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    config = variant_config(request.variant)
    data_yaml = validate_data_yaml(request.data)
    project = Path(request.project).expanduser().resolve()
    name = request.name or config["name"]
    run_dir = project / name
    version = require_ultralytics_version()
    trainer = _sgta_trainer_class() if config["uses_sgta"] else None
    initialization_manifest = None

    if request.total_epochs <= request.stage_epochs:
        raise ValueError("total_epochs must be greater than stage_epochs.")

    if request.resume:
        candidate = (
            Path(request.resume_checkpoint).expanduser()
            if request.resume_checkpoint
            else run_dir / "weights" / STAGE_CHECKPOINT
        )
        checkpoint = validate_resume_checkpoint(candidate, request.total_epochs)
        import torch

        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        completed_epochs = int(checkpoint_payload["epoch"]) + 1
        model = YOLO(str(checkpoint), task="detect")
        resume_source = str(checkpoint)
        if completed_epochs < request.stage_epochs:
            # An interrupted pre-stage run must still stop at the formal
            # screening boundary and preserve stage80_resume.pt.
            _attach_stage_callbacks(model, run_dir, request.stage_epochs)
    else:
        prepare_new_run_directory(run_dir)
        initialization_path = project / "_initialization" / f"{name}_init.pt"
        initialization_manifest = save_initialized_model(
            request.variant,
            weights=request.weights,
            output=initialization_path,
            seed=request.seed,
        )
        write_json(initialization_path.with_suffix(".json"), initialization_manifest)
        model = YOLO(str(initialization_path), task="detect")
        resume_source = None
        _attach_stage_callbacks(model, run_dir, request.stage_epochs)

    metadata = {
        **asdict(request),
        "name": name,
        "data": str(data_yaml),
        "project": str(project),
        "run_dir": str(run_dir),
        "resume_source": resume_source,
        "stage_checkpoint": str(run_dir / "weights" / STAGE_CHECKPOINT),
        "ultralytics_version": version,
        "git_commit": git_commit(),
        "initialization": initialization_manifest,
        "training_process": "direct current-process Python call",
        "scheduler_horizon": request.total_epochs,
    }
    write_json(run_dir / METADATA_FILE, metadata)
    print(f"Experiment: {name}")
    print(f"Variant: {request.variant}")
    print(f"Data: {data_yaml}")
    print(f"Run directory: {run_dir}")
    print(f"Stage/total epochs: {request.stage_epochs}/{request.total_epochs}")
    print(f"Resume: {request.resume}")
    if initialization_manifest is not None:
        transfer = initialization_manifest["weight_transfer"]
        print(
            f"Loaded state tensors: "
            f"{transfer['loaded_state_tensors']}/{transfer['total_state_tensors']}"
        )
        print(
            f"Loaded target parameter elements: "
            f"{transfer['loaded_target_parameter_elements']}/"
            f"{transfer['total_target_parameter_elements']}"
        )

    if request.resume:
        kwargs = {"resume": True}
    else:
        kwargs = {
            "data": str(data_yaml),
            # The scheduler is configured for 150 from the start. A callback
            # stops at 80 and preserves an unstripped resumable checkpoint.
            "epochs": request.total_epochs,
            "imgsz": request.imgsz,
            "batch": request.batch,
            "workers": request.workers,
            "device": request.device,
            "seed": request.seed,
            "deterministic": True,
            "optimizer": request.optimizer,
            "project": str(project),
            "name": name,
            "exist_ok": True,
        }
    if trainer is not None:
        kwargs["trainer"] = trainer
    result = model.train(**kwargs)
    summary = {
        "experiment_name": name,
        "variant": request.variant,
        "resume": request.resume,
        "resume_source": resume_source,
        "results": getattr(result, "results_dict", None) or {},
        "ultralytics_version": version,
        "git_commit": git_commit(),
    }
    write_json(run_dir / "summary.json", summary)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--name")
    parser.add_argument("--total-epochs", type=int, default=150)
    parser.add_argument("--stage-epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--resume", type=parse_bool, default=False)
    parser.add_argument("--resume-checkpoint")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    return run_training(
        TrainingRequest(
            variant=args.variant,
            data=args.data,
            project=args.project,
            weights=args.weights,
            name=args.name,
            total_epochs=args.total_epochs,
            stage_epochs=args.stage_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            optimizer=args.optimizer,
            resume=args.resume,
            resume_checkpoint=args.resume_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
