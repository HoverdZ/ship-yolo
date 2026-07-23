"""Direct-process formal training entrypoint for SPDDown and P2 Gaussian auxiliary experiments."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.spddown_p2aux_utils import (
    git_commit,
    require_ultralytics_version,
    save_initialized_model,
    variant_config,
    write_json,
)


METADATA_FILE = "experiment_metadata.json"


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
    epochs: int = 150
    imgsz: int = 640
    batch: int = 8
    workers: int = 2
    device: str = "0"
    seed: int = 0
    optimizer: str = "auto"
    resume: bool = False


def validate_data_yaml(path: str | Path) -> Path:
    data = Path(path).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}; no fallback is allowed.")
    return data


def validate_resume_checkpoint(path: str | Path) -> Path:
    import torch

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint type: {type(payload).__name__}")
    missing = [key for key in ("epoch", "optimizer", "train_args") if payload.get(key) is None]
    if missing:
        raise RuntimeError(f"Invalid resume checkpoint {checkpoint}; missing {missing}.")
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
            "Resume from weights/last.pt or choose another name."
        )


def _p2_gaussian_trainer_class():
    from ultralytics.models.yolo.detect import DetectionTrainer

    from custom_modules.p2_gaussian_aux import P2GaussianDetectionModel

    class P2GaussianTrainer(DetectionTrainer):
        """Use the custom loss model while retaining the official training loop."""

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                P2GaussianDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                )
            )
            if weights:
                model.load(weights)
            return model

        def get_validator(self):
            validator = super().get_validator()
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "p2_aux_loss"
            return validator

    return P2GaussianTrainer


def run_training(request: TrainingRequest):
    """Run training synchronously in the current Python process."""

    from custom_modules.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    config = variant_config(request.variant)
    data_yaml = validate_data_yaml(request.data)
    project = Path(request.project).expanduser().resolve()
    name = request.name or config["name"]
    run_dir = project / name
    version = require_ultralytics_version()

    trainer = _p2_gaussian_trainer_class() if request.variant == "p2_gaussian_aux" else None
    initialization_manifest = None
    if request.resume:
        checkpoint = validate_resume_checkpoint(run_dir / "weights" / "last.pt")
        model = YOLO(str(checkpoint), task="detect")
        resume_source = str(checkpoint)
    else:
        prepare_new_run_directory(run_dir)
        initialization_dir = project / "_initialization"
        initialization_path = initialization_dir / f"{name}_init.pt"
        initialization_manifest = save_initialized_model(
            request.variant,
            weights=request.weights,
            output=initialization_path,
            seed=request.seed,
        )
        write_json(initialization_path.with_suffix(".json"), initialization_manifest)
        model = YOLO(str(initialization_path), task="detect")
        resume_source = None

    metadata = {
        **asdict(request),
        "name": name,
        "data": str(data_yaml),
        "project": str(project),
        "run_dir": str(run_dir),
        "resume_source": resume_source,
        "ultralytics_version": version,
        "git_commit": git_commit(),
        "initialization": initialization_manifest,
        "training_process": "direct current-process Python call",
    }
    write_json(run_dir / METADATA_FILE, metadata)
    print(f"Experiment: {name}")
    print(f"Variant: {request.variant}")
    print(f"Data: {data_yaml}")
    print(f"Run directory: {run_dir}")
    print(f"Ultralytics: {version}")
    print(f"Git commit: {metadata['git_commit']}")
    print(f"Resume: {request.resume}")
    if initialization_manifest is not None:
        transfer = initialization_manifest["weight_transfer"]
        print(
            "Loaded state tensors: "
            f"{transfer['loaded_state_tensors']}/{transfer['total_state_tensors']}"
        )
        print(
            "Loaded target parameter elements: "
            f"{transfer['loaded_target_parameter_elements']}/"
            f"{transfer['total_target_parameter_elements']}"
        )

    if request.resume:
        kwargs = {"resume": True}
    else:
        kwargs = {
            "data": str(data_yaml),
            "epochs": request.epochs,
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
        "results": getattr(result, "results_dict", None) or {},
        "best": str(getattr(result, "best", "")),
        "last": str(getattr(result, "last", "")),
        "ultralytics_version": version,
        "git_commit": git_commit(),
    }
    write_json(run_dir / "summary.json", summary)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("spddown", "p2_gaussian_aux"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--resume", type=parse_bool, default=False)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    return run_training(TrainingRequest(**vars(args)))


if __name__ == "__main__":
    main()
