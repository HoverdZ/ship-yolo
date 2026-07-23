"""Colab/L4 entrypoint for one formal ASCGD experiment; importing never trains."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ascgd_utils import (
    CONFIG_PATH,
    DEFAULT_WEIGHTS,
    VARIANTS,
    audit_dataset,
    build_model,
    forward_signature,
    git_commit,
    model_statistics,
    read_yaml,
    register_modules,
    resolve_pretrained_weights,
    runtime_versions,
    save_initialized_model,
    sha256_file,
    variant_config,
    write_json,
)


def _run_directory(project: str | Path, name: str) -> Path:
    return Path(project).expanduser().resolve() / name


def _validate_run_state(run_dir: Path, resume: bool) -> Path | None:
    last = run_dir / "weights" / "last.pt"
    if resume:
        if not last.is_file():
            raise FileNotFoundError(
                f"--resume requires an existing checkpoint: {last}"
            )
        return last
    if run_dir.exists():
        if last.is_file():
            raise FileExistsError(
                f"Run already has last.pt: {run_dir}. Re-run with --resume."
            )
        raise FileExistsError(
            f"Run directory exists without last.pt and will not be overwritten: {run_dir}. "
            "Inspect it or choose a different --name."
        )
    return None


def _print_dataset_audit(audit: dict[str, Any]) -> None:
    print(f"Dataset: {audit['data_yaml']} (nc={audit['nc']}, names={audit['names']})")
    for split, values in audit["splits"].items():
        print(
            f"  {split}: images={values['images']} labels={values['labels']} "
            f"expected={values['expected_images']} difference={values['difference']}"
        )
    if not audit["expected_counts_match"]:
        print(
            "WARNING: dataset counts differ from 2582/842/874. The current data.yaml "
            "and resolved directories are being used; the difference is recorded."
        )


def _prepare_initialization(
    *,
    variant: str,
    source: Path,
    source_is_inception: bool,
    init_dir: Path,
    name: str,
) -> tuple[Path, dict[str, Any]]:
    init_dir.mkdir(parents=True, exist_ok=True)
    init_pt = init_dir / f"{name}_init.pt"
    manifest_path = init_dir / f"{name}_init.json"
    if init_pt.exists() or manifest_path.exists():
        if not init_pt.is_file() or not manifest_path.is_file():
            raise FileExistsError(
                f"Incomplete initialization pair: {init_pt}, {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks = {
            "variant": manifest.get("variant") == variant,
            "source": manifest.get("transfer", {}).get("source_sha256")
            == sha256_file(source),
            "checkpoint": manifest.get("output_sha256") == sha256_file(init_pt),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"Existing initialization does not match this request: {checks}. "
                "Use a different --name or --init-dir."
            )
        print(f"Reusing verified initialization: {init_pt}")
        return init_pt, manifest

    manifest = save_initialized_model(
        variant,
        source,
        init_pt,
        source_is_inception=source_is_inception,
        seed=0,
    )
    write_json(manifest_path, manifest)
    return init_pt, manifest


def _validation_summary(metrics: Any) -> dict[str, Any]:
    return {
        "results_dict": getattr(metrics, "results_dict", None) or {},
        "speed": getattr(metrics, "speed", None) or {},
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }


def _install_trainer_initialization_verifier(model: Any) -> None:
    """Abort before the first optimizer step if Trainer did not load the .pt."""

    expected = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.model.named_parameters()
    }

    def verify(trainer: Any) -> None:
        actual = dict(trainer.model.named_parameters())
        missing = sorted(set(expected) - set(actual))
        shape_mismatches = {
            name: {
                "expected": list(expected[name].shape),
                "actual": list(actual[name].shape),
            }
            for name in expected.keys() & actual.keys()
            if expected[name].shape != actual[name].shape
        }
        value_mismatches = [
            name
            for name in expected.keys() & actual.keys()
            if expected[name].shape == actual[name].shape
            and not torch.equal(
                expected[name],
                actual[name].detach().cpu().float(),
            )
        ]
        report = {
            "all_checks_passed": not missing
            and not shape_mismatches
            and not value_mismatches,
            "expected_parameter_tensors": len(expected),
            "actual_parameter_tensors": len(actual),
            "missing": missing,
            "shape_mismatches": shape_mismatches,
            "value_mismatches": value_mismatches,
        }
        write_json(
            Path(trainer.save_dir) / "initialization_verification.json",
            report,
        )
        if not report["all_checks_passed"]:
            raise RuntimeError(
                "Trainer initialization differs from the audited checkpoint: "
                f"{report}"
            )
        print(
            "Trainer initialization verified before optimizer step: "
            f"{len(expected)} parameter tensors"
        )

    model.add_callback("on_pretrain_routine_end", verify)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="e_full")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/ship_detection/organized_experiments"
        ),
    )
    parser.add_argument("--name")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--init-from-inception-best",
        type=Path,
        help="Debug only; formal experiments must omit this option.",
    )
    parser.add_argument("--init-dir", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = read_yaml(CONFIG_PATH)
    formal = config["training"]
    selected = variant_config(args.variant)
    name = args.name or selected["name"]
    project = args.project.expanduser().resolve()
    run_dir = _run_directory(project, name)
    last_pt = _validate_run_state(run_dir, args.resume)

    wants_cuda = str(args.device).lower() not in {"cpu", "none"}
    versions = runtime_versions()
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "Formal ASCGD training requested CUDA, but torch.cuda.is_available() is false."
        )
    print(json.dumps({"runtime": versions, "device": args.device}, indent=2))
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = audit_dataset(
        args.data,
        expected=config["dataset_expected_images"],
        require_single_class=True,
    )
    _print_dataset_audit(dataset)
    if args.epochs != formal["epochs"]:
        print(
            f"WARNING: epochs={args.epochs} differs from formal protocol "
            f"{formal['epochs']}."
        )
    if args.imgsz != formal["imgsz"] or args.batch != formal["batch"]:
        print("WARNING: image size or batch differs from the formal comparison protocol.")

    register_modules()
    transfer_manifest: dict[str, Any] | None = None
    if args.resume:
        from ultralytics import YOLO

        assert last_pt is not None
        model = YOLO(str(last_pt))
        print(f"Resuming exact optimizer/scheduler state from {last_pt}")
    else:
        source = resolve_pretrained_weights(
            args.init_from_inception_best or args.weights
        )
        source_is_inception = args.init_from_inception_best is not None
        if source_is_inception:
            print(
                "WARNING: using --init-from-inception-best debug initialization; "
                "this is not the formal fair comparison."
            )
        init_dir = (
            args.init_dir.expanduser().resolve()
            if args.init_dir
            else project / "_ascgd_initialization"
        )
        init_pt, transfer_manifest = _prepare_initialization(
            variant=args.variant,
            source=source,
            source_is_inception=source_is_inception,
            init_dir=init_dir,
            name=name,
        )
        from ultralytics import YOLO

        model = YOLO(str(init_pt))
        preflight_model = build_model(args.variant)
        preflight = forward_signature(
            preflight_model,
            imgsz=args.imgsz,
            batch=1,
            device="cpu",
        )
        statistics = model_statistics(preflight_model, imgsz=args.imgsz)
        if preflight["detect_spatial_sizes"] != [
            [args.imgsz // 8, args.imgsz // 8],
            [args.imgsz // 16, args.imgsz // 16],
            [args.imgsz // 32, args.imgsz // 32],
        ]:
            raise RuntimeError(f"Pre-training feature-shape check failed: {preflight}")
        print(
            json.dumps(
                {
                    "model_yaml": selected["yaml"],
                    "statistics": statistics,
                    "preflight": preflight,
                    "weight_transfer": {
                        key: transfer_manifest["transfer"][key]
                        for key in (
                            "inherited_parameter_elements",
                            "target_parameter_elements",
                            "parameter_element_inheritance_ratio",
                            "backbone_parameter_inheritance_ratio",
                            "detect_parameter_inheritance_ratio",
                        )
                    },
                },
                indent=2,
            )
        )

    _install_trainer_initialization_verifier(model)
    request_metadata = {
        "variant": args.variant,
        "experiment_name": name,
        "model_yaml": selected["yaml"],
        "data": str(args.data.expanduser().resolve()),
        "project": str(project),
        "run_dir": str(run_dir),
        "runtime": versions,
        "dataset_audit": dataset,
        "training_protocol": formal,
        "requested": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "resume": args.resume,
        },
        "git_commit": git_commit(),
        "initialization": transfer_manifest,
    }
    staging = project / "_ascgd_initialization" / f"{name}_training_request.json"
    write_json(staging, request_metadata)

    if args.resume:
        results = model.train(resume=True)
    else:
        augment = formal["augmentations"]
        results = model.train(
            data=str(args.data.expanduser().resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=formal["seed"],
            deterministic=formal["deterministic"],
            optimizer=formal["optimizer"],
            lr0=formal["lr0"],
            lrf=formal["lrf"],
            momentum=formal["momentum"],
            weight_decay=formal["weight_decay"],
            warmup_epochs=formal["warmup_epochs"],
            warmup_momentum=formal["warmup_momentum"],
            warmup_bias_lr=formal["warmup_bias_lr"],
            patience=formal["patience"],
            close_mosaic=formal["close_mosaic"],
            amp=formal["amp"],
            cache=formal["cache"],
            val=formal["val"],
            plots=formal["plots"],
            single_cls=formal["single_class"],
            hsv_h=augment["hsv_h"],
            hsv_s=augment["hsv_s"],
            hsv_v=augment["hsv_v"],
            degrees=augment["degrees"],
            translate=augment["translate"],
            scale=augment["scale"],
            shear=augment["shear"],
            perspective=augment["perspective"],
            flipud=augment["flipud"],
            fliplr=augment["fliplr"],
            mosaic=augment["mosaic"],
            mixup=augment["mixup"],
            copy_paste=augment["copy_paste"],
            project=str(project),
            name=name,
            exist_ok=False,
        )

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Training finished without best.pt: {best_pt}")
    from ultralytics import YOLO

    best_model = YOLO(str(best_pt))
    validation = best_model.val(
        data=str(args.data.expanduser().resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        split="val",
        project=str(run_dir),
        name="best_validation",
        exist_ok=False,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, run_dir / "ascgd_experiments.yaml")
    write_json(run_dir / "ascgd_training_request.json", request_metadata)
    summary = {
        "variant": args.variant,
        "experiment_name": name,
        "git_commit": git_commit(),
        "best_pt": str(best_pt),
        "last_pt": str(run_dir / "weights" / "last.pt"),
        "train_results": getattr(results, "results_dict", None) or {},
        "best_validation": _validation_summary(validation),
        "runtime": runtime_versions(),
        "formal_training_completed": True,
    }
    write_json(run_dir / "ascgd_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
