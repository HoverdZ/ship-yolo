from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from tools.paper_artifacts.formal_protocol import (
    EXPERIMENTS,
    AtomicDriveMirror,
    FormalConfig,
    audit_dataset,
    audit_model,
    build_and_initialize,
    copy_dataset_to_local,
    restore_or_guard_run,
    train_foreground,
)
from tools.paper_artifacts.compare_per_image_predictions import compare
from tools.paper_artifacts.export_experiment_bundle import export_bundle
from tools.paper_artifacts.select_visual_examples import select
from tools.paper_artifacts.summarize_ablation import summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("experiment_id", list(EXPERIMENTS))
def test_formal_model_builds_with_exact_stride_and_detect_inputs(experiment_id: str, tmp_path: Path) -> None:
    config = FormalConfig(
        experiment_id=experiment_id,
        local_runs_root=str(tmp_path / "runs"),
        local_data_root=str(tmp_path / "dataset"),
    )
    config.local_yaml.parent.mkdir(parents=True, exist_ok=True)
    config.local_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "x", "val": "x", "test": "x", "nc": 1, "names": {0: "ship"}}),
        encoding="utf-8",
    )
    model, transfer = build_and_initialize(config, ROOT / "yolo11n.pt")
    assert transfer["passed"]
    assert model.ckpt and model.ckpt["formal_pretrained_transfer"] is True
    assert [float(value) for value in model.model.stride] == config.spec["strides"]
    assert list(model.model.model[-1].f) == config.spec["detect_from"]


def test_vgup_transfer_counts_only_real_official_tensors(tmp_path: Path) -> None:
    config = FormalConfig(
        experiment_id="A4_inceptiondw_dpls_scam_vgup",
        local_runs_root=str(tmp_path / "runs"),
        local_data_root=str(tmp_path / "dataset"),
    )
    config.local_yaml.parent.mkdir(parents=True, exist_ok=True)
    config.local_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "x", "val": "x", "test": "x", "nc": 1, "names": {0: "ship"}}),
        encoding="utf-8",
    )
    _model, transfer = build_and_initialize(config, ROOT / "yolo11n.pt")
    assert transfer["method"] == "official_same_shape_with_input_layer_shift"
    assert transfer["loaded_tensors"] < transfer["target_state_tensors"]
    assert all(not key.startswith("model.0.") for key in transfer["sample_mapping"])
    assert "official_to_A3" not in transfer


def test_dataset_yaml_parent_fallback_audit_and_copy_are_read_only(tmp_path: Path) -> None:
    source = tmp_path / "drive" / "ship_detection" / "data"
    for split in ("train", "val", "test"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir(parents=True)
        Image.new("RGB", (16, 16), "black").save(source / split / "images" / "sample.jpg")
        (source / split / "labels" / "sample.txt").write_text("" if split == "val" else "0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data_yaml = source / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": "/content/ship_detection/data",
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "ship"},
            }
        ),
        encoding="utf-8",
    )
    before = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    config = FormalConfig(
        experiment_id="A0_yolo11n",
        drive_data_yaml=str(data_yaml),
        local_data_root=str(tmp_path / "local"),
        local_runs_root=str(tmp_path / "runs"),
        copy_workers=2,
    )
    report = audit_dataset(config)
    local_yaml = copy_dataset_to_local(config)
    after = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert before == after
    assert report["resolved_root"] == str(source.resolve())
    assert report["splits"]["val"]["empty_labels"] == 1
    assert yaml.safe_load(local_yaml.read_text(encoding="utf-8"))["path"] == str(tmp_path / "local")


def test_dataset_copy_reports_progress_in_completion_order() -> None:
    from tools.paper_artifacts.formal_protocol import copy_dataset_to_local

    source = inspect.getsource(copy_dataset_to_local)
    assert "concurrent.futures.as_completed" in source
    assert "pool.map(" not in source
    assert "Reading source sizes" not in source
    assert 'desc="Processed bytes"' in source
    assert "file=sys.stdout" in source
    assert "flush=True" in source


def test_cpu_audit_does_not_mutate_weights_buffers_or_leave_gradients(tmp_path: Path) -> None:
    config = FormalConfig(
        experiment_id="A0_yolo11n",
        local_runs_root=str(tmp_path / "runs"),
        local_data_root=str(tmp_path / "dataset"),
    )
    config.local_yaml.parent.mkdir(parents=True, exist_ok=True)
    config.local_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "x", "val": "x", "test": "x", "nc": 1, "names": {0: "ship"}}),
        encoding="utf-8",
    )
    model, _transfer = build_and_initialize(config, ROOT / "yolo11n.pt")
    before = {key: value.detach().cpu().clone() for key, value in model.model.state_dict().items()}
    audit_model(config, model, backward_imgsz=64)
    after = model.model.state_dict()
    assert all(torch.equal(before[key], after[key].detach().cpu()) for key in before)
    assert all(parameter.grad is None for parameter in model.model.parameters())


def test_resume_guards_cross_experiment_and_completed_runs(tmp_path: Path) -> None:
    config = FormalConfig(
        experiment_id="A0_yolo11n",
        local_runs_root=str(tmp_path / "runs"),
        drive_experiment_root=str(tmp_path / "drive"),
    )
    assert restore_or_guard_run(config) == "new"
    (config.run_dir / "weights").mkdir(parents=True)
    (config.run_dir / "weights" / "last.pt").write_bytes(b"checkpoint")
    (config.run_dir / "experiment_state.json").write_text(json.dumps({"experiment_id": config.experiment_id}), encoding="utf-8")
    assert restore_or_guard_run(config) == "resume"
    (config.run_dir / "experiment_state.json").write_text(json.dumps({"experiment_id": "A1_inceptiondw"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cross-experiment"):
        restore_or_guard_run(config)
    (config.run_dir / "COMPLETED.ok").write_text("done", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already complete"):
        restore_or_guard_run(config)


def test_run_id_selects_an_independent_directory_and_guards_resume(tmp_path: Path) -> None:
    config = FormalConfig(
        experiment_id="A0_yolo11n",
        run_id="A0_yolo11n_v2",
        local_runs_root=str(tmp_path / "runs"),
        drive_experiment_root=str(tmp_path / "drive"),
    )
    assert config.run_dir.name == "A0_yolo11n_v2"
    assert config.drive_dir.name == "A0_yolo11n_v2"
    (config.run_dir / "weights").mkdir(parents=True)
    (config.run_dir / "weights" / "last.pt").write_bytes(b"checkpoint")
    (config.run_dir / "experiment_state.json").write_text(
        json.dumps({"experiment_id": config.experiment_id, "run_id": "wrong_run"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cross-run"):
        restore_or_guard_run(config)


def test_atomic_mirror_replaces_complete_files(tmp_path: Path) -> None:
    local = tmp_path / "local"
    drive = tmp_path / "drive"
    local.mkdir()
    (local / "results.csv").write_text("epoch,value\n0,1\n", encoding="utf-8")
    mirror = AtomicDriveMirror(local, drive)
    mirror.enqueue("results.csv")
    mirror.close()
    assert (drive / "results.csv").read_text(encoding="utf-8") == "epoch,value\n0,1\n"
    assert not list(drive.rglob("*.tmp"))


def test_training_is_direct_and_protocol_is_staged() -> None:
    source = inspect.getsource(train_foreground)
    assert "model.train(" in source
    assert "subprocess" not in source
    assert "exist_ok=False" in source
    assert "on_pretrain_routine_start" in source
    assert "on_pretrain_routine_end" in source
    assert "pretrained=True" in source
    assert "pretrained=False" not in source
    assert "_verify_trainer_handoff" in source


def test_specialty_artifact_code_covers_required_module_evidence() -> None:
    from tools.paper_artifacts import generate_specialty_artifacts as specialty

    source = inspect.getsource(specialty)
    for required in (
        "P{level}_local_contrast.png",
        "P{level}_contrast_gate.png",
        "P{level}_context_residual.png",
        "P{level}_scam_input_energy.png",
        "P{level}_scam_output_energy.png",
        "vgup_luminance_histogram.png",
        "vgup_spatial_visibility_gate.png",
        "input_reference.png",
    ):
        assert required in source


def test_formal_notebooks_have_unique_ids_and_no_training_subprocess() -> None:
    notebooks = sorted((ROOT / "colab" / "formal_ablation_v1").glob("*.ipynb"))
    assert len(notebooks) == 6
    ids = set()
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        for experiment_id in EXPERIMENTS:
            if f'EXPERIMENT_ID = "{experiment_id}"' in text:
                ids.add(experiment_id)
        training = [cell for cell in payload["cells"] if "formal-training" in cell.get("metadata", {}).get("tags", [])]
        assert len(training) == 1
        training_text = "".join(training[0]["source"])
        assert "train_foreground(" in training_text
        assert "subprocess" not in training_text
        assert 'LOCAL_RUNS_ROOT = "/content/formal_runs_v2"' in text
        assert 'DRIVE_EXPERIMENT_ROOT = "/content/drive/MyDrive/ShipPaper/formal_ablation_v2"' in text
    assert ids == set(EXPERIMENTS)


def test_cross_run_summary_selection_and_bundle_tools(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    for index, (experiment_id, spec) in enumerate(EXPERIMENTS.items()):
        run = root / experiment_id
        run.mkdir(parents=True)
        metrics = {
            "precision": 0.80 + index * 0.001,
            "recall": 0.70 + index * 0.001,
            "map50": 0.77 + index * 0.001,
            "map75": 0.30 + index * 0.001,
            "map50_95": 0.32 + index * 0.001,
        }
        (run / "run_manifest.json").write_text(
            json.dumps({"best_metrics": metrics}),
            encoding="utf-8",
        )
        (run / "complexity.json").write_text(
            json.dumps(
                {
                    "parameters": 2_500_000 + index,
                    "gflops": 6.5 + index * 0.1,
                    "model_size_bytes": 5_000_000 + index,
                    "pytorch_fp32": {"mean_ms": 5.0 + index},
                }
            ),
            encoding="utf-8",
        )
        (run / "best_epoch_summary.json").write_text(
            json.dumps({"best_epoch": 100 + index, "training_time_seconds": 3600 + index}),
            encoding="utf-8",
        )
        (run / "val_image_metrics.csv").write_text(
            "image,tp,fp,fn,precision,recall\n"
            f"val/images/a.jpg,1,{index % 2},{1 if index < 2 else 0},0.5,0.5\n",
            encoding="utf-8",
        )
    frame = summarize(root)
    assert len(frame) == 6
    assert (
        (root / "paper_summary" / "formal_ablation_results.xlsx").is_file()
        or (root / "paper_summary" / "formal_ablation_results.xlsx.unavailable.txt").is_file()
    )
    assert len(compare(root)) == 1
    assert not select(root).empty
    bundle = export_bundle(root / "A0_yolo11n", tmp_path / "A0.zip")
    assert bundle.is_file() and bundle.stat().st_size > 0
