"""Formal YOLO11n-InceptionDW training entrypoint; this file does not auto-run training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.inceptiondw_utils import (
    EXPERIMENT_NAME,
    MODEL_YAML,
    build_custom_model,
    model_statistics,
    require_ultralytics_version,
    structure_report,
    transfer_pretrained_weights,
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


def git_commit() -> str:
    """Resolve HEAD directly from the repository metadata."""

    git_path = ROOT / ".git"
    try:
        if git_path.is_file():
            git_dir = (ROOT / git_path.read_text(encoding="utf-8").split(":", 1)[1].strip()).resolve()
        else:
            git_dir = git_path
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref_name = head.removeprefix("ref: ")
        loose_ref = git_dir / ref_name
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="utf-8").strip()
        packed_refs = git_dir / "packed-refs"
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref_name:
                    return commit
    except (OSError, IndexError, ValueError):
        pass
    return "unknown"


def validate_data_yaml(path: str | Path) -> Path:
    data = Path(path).expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data}. Refusing to fall back to a built-in dataset."
        )
    return data


def validate_resume_checkpoint(path: str | Path) -> Path:
    import torch

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid resume checkpoint type: {type(payload).__name__}")
    missing = [
        key
        for key in ("epoch", "optimizer", "train_args")
        if key not in payload or payload[key] is None
    ]
    if missing:
        raise RuntimeError(f"Invalid resume checkpoint {checkpoint}; missing {missing}.")
    return checkpoint


def prepare_new_run_directory(run_dir: Path) -> None:
    """Allow a fresh directory or this script's metadata-only staging directory."""

    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"Run path is not a directory: {run_dir}")
    entries = {entry.name for entry in run_dir.iterdir()}
    if entries and not entries.issubset({METADATA_FILE}):
        raise FileExistsError(
            f"Run directory already contains training artifacts: {run_dir}. "
            "Use --resume true with its weights/last.pt or choose another --name."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_YAML))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--data", required=True, help="Dataset YAML; no implicit fallback.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default=EXPERIMENT_NAME)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--resume", type=parse_bool, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_yaml = validate_data_yaml(args.data)
    project = Path(args.project).expanduser().resolve()
    run_dir = project / args.name
    version = require_ultralytics_version()

    # Build a fresh target to report the formal initialization even when a
    # resumed run will subsequently load last.pt.
    initialization_model = build_custom_model(args.model)
    structure = structure_report(initialization_model)
    if not structure["all_checks_passed"]:
        raise RuntimeError(f"Structure validation failed: {structure}")
    transfer = transfer_pretrained_weights(
        initialization_model,
        args.weights,
        apply=not args.resume,
    )
    stats = model_statistics(initialization_model, imgsz=args.imgsz)

    if args.resume:
        last_pt = validate_resume_checkpoint(run_dir / "weights" / "last.pt")
        from ultralytics import YOLO

        model = YOLO(str(last_pt))
        resume_source = str(last_pt)
    else:
        prepare_new_run_directory(run_dir)
        model = initialization_model
        resume_source = None

    metadata = {
        "experiment_name": args.name,
        "model_yaml": str(Path(args.model).expanduser().resolve()),
        "weights": str(args.weights),
        "data": str(data_yaml),
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
        "ultralytics_version": version,
        "git_commit": git_commit(),
        "model_statistics": stats,
        "weight_transfer": transfer,
        "structure": structure,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / METADATA_FILE, metadata)

    print("YOLO11n-InceptionDW formal training request")
    print(f"  model: {metadata['model_yaml']}")
    print(f"  data: {data_yaml}")
    print(f"  official weights: {args.weights}")
    print(f"  run directory: {run_dir}")
    print(f"  parameters: {stats['parameters']}")
    print(f"  GFLOPs: {stats['gflops']:.6f}")
    print(
        "  inherited parameter elements: "
        f"{transfer['inherited_parameter_elements']}/{transfer['target_parameter_elements']} "
        f"({transfer['parameter_element_inheritance_ratio']:.4%})"
    )
    print(f"  resume: {args.resume}")

    if args.resume:
        result = model.train(resume=True)
    else:
        result = model.train(
            data=str(data_yaml),
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
        "results": getattr(result, "results_dict", None) or {},
        "best": str(getattr(result, "best", "")),
        "last": str(getattr(result, "last", "")),
        "ultralytics_version": version,
        "git_commit": git_commit(),
    }
    write_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
