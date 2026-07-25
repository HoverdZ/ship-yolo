"""Tests for the shallow C3Cross P2/P3 screening workflow."""

from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2, Conv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.c3k2_crossconv import C3k2CrossConv
from custom_modules.register import register_module_ablation_modules
from tools.c3cross_p23_workflow import (
    CROSS_PREFIXES,
    MODEL_YAML,
    build_p23_model,
    cpu_forward_backward_report,
    hybrid_initialize,
    make_epoch15_screen_callback,
    structure_report,
    validate_checkpoint_metrics,
)
import colab.train_yolo11n_c3cross_p23 as colab_helpers


def _fill_state(model: YOLO, floating: float, integer: int) -> None:
    with torch.no_grad():
        for tensor in model.model.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(floating)
            else:
                tensor.fill_(integer)


def test_structure_preserves_stride2_conv_and_only_replaces_p2_p3() -> None:
    model = build_p23_model()
    layers = model.model.model
    report = structure_report(model)
    assert report["passed"], report
    assert report["crossconv_indices"] == [2, 4]
    for index in (1, 3):
        assert isinstance(layers[index], Conv)
        assert layers[index].conv.kernel_size == (3, 3)
        assert layers[index].conv.stride == (2, 2)
    for index in (6, 8):
        assert isinstance(layers[index], C3k2)
        assert not isinstance(layers[index], C3k2CrossConv)


def test_cpu_forward_backward_smoke_is_finite() -> None:
    report = cpu_forward_backward_report(build_p23_model(), imgsz=64)
    assert report["passed"], report
    assert report["input_gradient_finite"]
    assert report["parameter_gradients"] > 0


def test_hybrid_initialization_has_exact_tensor_provenance(tmp_path: Path) -> None:
    register_module_ablation_modules()
    baseline = YOLO(str(ROOT / "yolo11n.pt"), verbose=False)
    c3cross = YOLO(str(ROOT / "experiments/yolo11n-c3cross.yaml"), verbose=False)
    _fill_state(baseline, floating=0.125, integer=3)
    _fill_state(c3cross, floating=0.75, integer=7)
    baseline_pt = tmp_path / "baseline.pt"
    c3cross_pt = tmp_path / "c3cross.pt"
    baseline.save(baseline_pt)
    c3cross.save(c3cross_pt)

    target, report = hybrid_initialize(baseline_pt, c3cross_pt)
    target_state = target.model.state_dict()
    baseline_state = baseline.model.state_dict()
    c3cross_state = c3cross.model.state_dict()

    assert report["passed"], report
    assert report["loaded_tensors"]["loaded_total"] == len(target_state)
    assert report["loaded_tensors"]["target_total"] == len(target_state)
    assert not report["missing_target_keys"]

    for key, value in target_state.items():
        if key.startswith(CROSS_PREFIXES):
            assert torch.equal(value.cpu(), c3cross_state[key].cpu()), key
        else:
            assert torch.equal(value.cpu(), baseline_state[key].cpu()), key


def _write_screen_csv(path: Path, *, map50_95: float, recall: float) -> None:
    pd.DataFrame(
        [
            {
                "epoch": epoch,
                "metrics/precision(B)": 0.80,
                "metrics/recall(B)": recall,
                "metrics/mAP50(B)": 0.77,
                "metrics/mAP50-95(B)": map50_95,
            }
            for epoch in range(1, 16)
        ]
    ).to_csv(path, index=False)


def test_epoch15_callback_stops_failed_run_and_records_decision(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    _write_screen_csv(csv_path, map50_95=0.319, recall=0.701)
    trainer = SimpleNamespace(
        epoch=14,
        csv=csv_path,
        save_dir=tmp_path,
        stop=False,
    )
    make_epoch15_screen_callback()(trainer)
    assert trainer.stop is True
    payload = json.loads(
        (tmp_path / "epoch15_screen.json").read_text(encoding="utf-8")
    )
    assert payload["continue_to_epoch_30"] is False


def test_epoch15_callback_continues_passing_run(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    _write_screen_csv(csv_path, map50_95=0.321, recall=0.701)
    trainer = SimpleNamespace(
        epoch=14,
        csv=csv_path,
        save_dir=tmp_path,
        stop=False,
    )
    make_epoch15_screen_callback()(trainer)
    assert trainer.stop is False
    payload = json.loads(
        (tmp_path / "epoch15_screen.json").read_text(encoding="utf-8")
    )
    assert payload["continue_to_epoch_30"] is True


def test_experiment_yaml_is_independent() -> None:
    assert MODEL_YAML.is_file()
    assert MODEL_YAML.name == "yolo11n-c3cross-p23.yaml"


def test_ap75_uses_ultralytics_map75_directly() -> None:
    source = inspect.getsource(validate_checkpoint_metrics)
    assert "metrics.box.map75" in source
    assert 'split="val"' in source
    assert "augment=False" in source


def test_colab_copy_uses_sixteen_copyfile_workers() -> None:
    source = inspect.getsource(colab_helpers)
    assert "COPY_WORKERS = 16" in source
    assert "shutil.copyfile(source, destination)" in source
    assert "ThreadPoolExecutor(max_workers=workers)" in source
    assert "executor.map(_copy_one, jobs, chunksize=8)" in source


def test_colab_notebook_is_valid_safe_and_targets_new_branch() -> None:
    notebook_path = ROOT / "colab/YOLO11n_C3Cross_P23_Screening.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "ultralytics==8.4.92" in text
    assert "experiment/yolo11n-c3cross-p23" in text
    assert "layer 1、3" in text
    assert "copy_dataset_to_local(workers=16)" in text
    assert "metrics.box.map75" in inspect.getsource(validate_checkpoint_metrics)
    assert "run_p23_screening(" in text
    assert "RUN_FINETUNE = False" in text
    assert "ghp_" not in text
    assert "github.com/" + "x-access-token" not in text
