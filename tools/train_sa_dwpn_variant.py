"""Unified SA-DWPN formal ablation training entrypoint."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.register import register_sa_dwpn_modules
from tools.sa_dwpn_utils import (
    PROTOCOL_PATH,
    ensure_data_yaml,
    git_commit,
    load_protocol,
    protocol_hash,
    shape_matched_transfer_report,
    validate_resume_checkpoint,
    variant_config,
    write_json,
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SA-DWPN variant with the unified protocol.")
    parser.add_argument("--variant", required=True, choices=["b", "c_lite", "t3_only", "o3_only"])
    parser.add_argument("--data", required=True, help="Dataset YAML. Required; no coco8 fallback is allowed.")
    parser.add_argument("--weights", default="", help="Official yolo11n.pt path. Defaults to protocol value.")
    parser.add_argument("--project", required=True, help="Output project directory.")
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    parser.add_argument("--resume", type=parse_bool, default=False)
    parser.add_argument("--exist-ok", action="store_true", help="Allow starting a new run in an existing directory.")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    train_cfg = protocol["training"]
    variant = variant_config(args.variant)
    data_yaml = ensure_data_yaml(args.data)
    weights = args.weights or protocol["initialization"]["official_weights"]
    run_dir = Path(args.project) / variant["name"]

    if args.resume:
        resume_path = run_dir / "weights" / "last.pt"
        resume_info = validate_resume_checkpoint(resume_path)
    else:
        resume_info = None
        if run_dir.exists() and not args.exist_ok:
            raise FileExistsError(f"Run directory already exists: {run_dir}. Use --resume true or --exist-ok explicitly.")

    register_sa_dwpn_modules()
    from ultralytics import YOLO
    import ultralytics

    model = YOLO(str(variant["yaml"]))
    transfer_report = shape_matched_transfer_report(model, weights)

    print("SA-DWPN training request")
    print(f"  variant: {args.variant}")
    print(f"  model YAML: {variant['yaml']}")
    print(f"  official weights: {weights}")
    print(f"  data YAML: {data_yaml}")
    print(f"  run directory: {run_dir}")
    print(f"  epochs: {train_cfg['epochs']}")
    print(f"  batch: {train_cfg['batch']}")
    print(f"  imgsz: {train_cfg['imgsz']}")
    print(f"  workers: {train_cfg['workers']}")
    print(f"  seed: {train_cfg['seed']}")
    print(f"  spatial positions: {variant['spatial_positions']}")
    print(f"  pretrained transfer report: {transfer_report}")

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_args = {
        "variant": args.variant,
        "experiment_name": variant["name"],
        "model_yaml": str(variant["yaml"]),
        "weights": str(weights),
        "data": str(data_yaml),
        "project": str(args.project),
        "run_dir": str(run_dir),
        "resume": args.resume,
        "resume_info": resume_info,
        "protocol": protocol,
        "protocol_hash": protocol_hash(args.protocol),
        "git_commit": git_commit(),
        "ultralytics_version": getattr(ultralytics, "__version__", "unknown"),
        "transfer_report": transfer_report,
    }
    write_json(run_dir / "resolved_args.json", resolved_args)
    shutil.copyfile(args.protocol, run_dir / "protocol.yaml")

    if args.resume:
        train_result = model.train(resume=True)
    else:
        model.load(weights)
        train_result = model.train(
            data=str(data_yaml),
            imgsz=train_cfg["imgsz"],
            epochs=train_cfg["epochs"],
            batch=train_cfg["batch"],
            device=train_cfg["device"],
            workers=train_cfg["workers"],
            seed=train_cfg["seed"],
            deterministic=train_cfg["deterministic"],
            amp=train_cfg["amp"],
            cache=train_cfg["cache"],
            patience=train_cfg["patience"],
            save=train_cfg["save"],
            save_period=train_cfg["save_period"],
            plots=train_cfg["plots"],
            verbose=train_cfg["verbose"],
            project=str(args.project),
            name=variant["name"],
            exist_ok=args.exist_ok,
        )

    metrics = getattr(train_result, "results_dict", None) or {}
    summary = {
        "experiment_name": variant["name"],
        "variant": args.variant,
        "metrics": metrics,
        "params": sum(p.numel() for p in model.model.parameters()),
        "gflops": getattr(model.model, "gflops", None),
        "git_commit": git_commit(),
        "ultralytics_version": getattr(ultralytics, "__version__", "unknown"),
        "protocol_hash": protocol_hash(args.protocol),
    }
    write_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
