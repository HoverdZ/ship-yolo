"""Formal direct-process training entrypoint; importing this file never trains."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from functools import partial
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fapn_prefusion_utils import (
    EXPECTED_ULTRALYTICS_VERSION,
    install_safe_prefusion_flops,
    prepare_formal_run_directory,
    register_modules,
    require_ultralytics_version,
    validate_init_manifest,
    variant_config,
    verify_prefusion_trainer_initialization,
)


DETERMINISTIC_DCN_WARNING = (
    r".*compute_grad_input does not have a deterministic implementation.*"
)


def validate_runtime(device: int | str) -> dict:
    version = require_ultralytics_version()
    wants_cuda = str(device).lower() not in {"cpu", "none"}
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable.")
    return {
        "ultralytics": version,
        "required_ultralytics": EXPECTED_ULTRALYTICS_VERSION,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_formal_training(
    variant: str,
    *,
    data: str | Path,
    init_pt: str | Path | None = None,
    manifest: str | Path | None = None,
    profile: str | Path | None = None,
    project: str | Path,
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 2,
    device: int | str = 0,
):
    """Run the official Ultralytics Python API from the real init.pt."""

    config = variant_config(variant)
    data = Path(data).expanduser().resolve()
    init_pt = Path(init_pt or config["init_pt"]).expanduser().resolve()
    manifest = Path(manifest or config["manifest"]).expanduser().resolve()
    profile = Path(profile or config["profile"]).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Local dataset YAML not found: {data}")
    runtime = validate_runtime(device)
    init_audit = validate_init_manifest(init_pt, manifest)
    if not init_audit["all_checks_passed"]:
        raise RuntimeError(f"Initialization manifest failed: {init_audit}")
    run_dir, backup = prepare_formal_run_directory(project, config["experiment_name"])
    print(json.dumps({"runtime": runtime, "run_dir": str(run_dir), "backup": str(backup)}, indent=2))

    register_modules()
    restore_flops = install_safe_prefusion_flops(profile)
    try:
        from ultralytics import YOLO

        model = YOLO(str(init_pt))
        verifier_callback = partial(
            verify_prefusion_trainer_initialization,
            manifest_path=manifest,
        )
        model.add_callback("on_pretrain_routine_end", verifier_callback)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=DETERMINISTIC_DCN_WARNING,
                category=UserWarning,
            )
            return model.train(
                data=str(data),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                seed=0,
                deterministic=True,
                optimizer="auto",
                amp=True,
                val=True,
                plots=True,
                patience=100,
                project=str(Path(project).expanduser().resolve()),
                name=config["experiment_name"],
                exist_ok=False,
            )
    finally:
        restore_flops()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "inceptiondw"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init-pt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run_formal_training(
        args.variant,
        data=args.data,
        init_pt=args.init_pt,
        manifest=args.manifest,
        profile=args.profile,
        project=args.project,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
