"""Pre-training tests for YOLO11n-InceptionDW-SCSharedHead."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.scshared_head import SCSharedDetect, ScaleCalibration
from tools.scshared_head_utils import (
    build_model,
    forward_report,
    structure_report,
    transfer_weights,
)
from tools.train_scshared_head import (
    STAGE_CHECKPOINT,
    _attach_stage_callbacks,
    validate_resume_checkpoint,
)


def test_scale_calibration_is_identity_and_positive() -> None:
    module = ScaleCalibration()
    sample = torch.randn(2, 8, 10, 10)
    assert torch.equal(module(sample), sample)
    with torch.no_grad():
        module.log_scale.fill_(-3.0)
    output = module(sample)
    assert torch.allclose(output, sample * torch.exp(torch.tensor(-3.0)))
    assert float(module.log_scale.detach().exp()) > 0


def test_shared_stem_is_reused_for_all_three_scales() -> None:
    head = SCSharedDetect(
        nc=1,
        shared_channels=64,
        gn_groups=16,
        ch=(64, 128, 256),
    ).train()
    calls = []
    hook = head.shared_stem[0].register_forward_hook(
        lambda *_: calls.append("shared")
    )
    try:
        output = head(
            [
                torch.randn(2, 64, 16, 16),
                torch.randn(2, 128, 8, 8),
                torch.randn(2, 256, 4, 4),
            ]
        )
    finally:
        hook.remove()
    assert len(calls) == 3
    assert list(output["boxes"].shape) == [2, 64, 336]
    assert list(output["scores"].shape) == [2, 1, 336]
    assert torch.isfinite(output["boxes"]).all()
    assert torch.isfinite(output["scores"]).all()


def test_model_structure_forward_and_weight_transfer() -> None:
    model = build_model()
    structure = structure_report(model)
    assert structure["all_checks_passed"]
    assert structure["scale_values"] == [1.0, 1.0, 1.0]

    forward = forward_report(imgsz=128)
    assert forward["all_checks_passed"]

    transfer = transfer_weights(model, ROOT / "yolo11n.pt")
    assert transfer["mapped_native_detect_output_tensors"] == 6
    assert transfer["loaded_target_parameter_element_ratio"] > 0.95
    assert not [
        key
        for key in transfer["random_initialized_non_head_parameter_keys"]
        if not key.startswith(
            (
                "model.2.m.0.cv2.",
                "model.2.m.0.cv2_adapter.",
                "model.4.m.0.cv2.",
                "model.4.m.0.cv2_adapter.",
            )
        )
    ]


def test_native_detection_loss_runs_and_backpropagates() -> None:
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

    network = build_model().model.cpu().train()
    network.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    batch = {
        "img": torch.randn(2, 3, 128, 128),
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.zeros((2, 1)),
        "bboxes": torch.tensor(
            [[0.50, 0.50, 0.05, 0.04], [0.30, 0.40, 0.10, 0.08]]
        ),
    }
    loss, items = network(batch)
    loss.sum().backward()
    gradient = network.model[-1].shared_stem[0].conv.weight.grad
    assert list(items.shape) == [3]
    assert torch.isfinite(loss).all()
    assert gradient is not None and torch.isfinite(gradient).all()


def test_stage_callback_preserves_raw_checkpoint(tmp_path: Path) -> None:
    callbacks = {}

    class DummyModel:
        def add_callback(self, event, callback):
            callbacks.setdefault(event, []).append(callback)

    class DummyTrainer:
        epoch = 79
        stop = False
        last = tmp_path / "run" / "weights" / "last.pt"

    trainer = DummyTrainer()
    trainer.last.parent.mkdir(parents=True)
    trainer.last.write_bytes(b"raw resumable checkpoint")
    _attach_stage_callbacks(DummyModel(), tmp_path / "run", 80)
    callbacks["on_train_epoch_end"][0](trainer)
    assert trainer.stop
    callbacks["on_model_save"][0](trainer)
    stage = trainer.last.parent / STAGE_CHECKPOINT
    assert stage.read_bytes() == trainer.last.read_bytes()


def test_resume_validation_requires_optimizer_and_150_epoch_horizon(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.pt"
    torch.save(
        {"epoch": 79, "optimizer": {"state": {}}, "train_args": {"epochs": 150}},
        valid,
    )
    assert validate_resume_checkpoint(valid, 150) == valid.resolve()

    invalid = tmp_path / "invalid.pt"
    torch.save(
        {"epoch": 79, "optimizer": None, "train_args": {"epochs": 80}},
        invalid,
    )
    with pytest.raises(RuntimeError, match="缺少续训状态"):
        validate_resume_checkpoint(invalid, 150)


def test_training_entry_is_direct_process() -> None:
    import tools.train_scshared_head as training

    source = inspect.getsource(training.run_training)
    assert "subprocess" not in source
    assert "model.train(" in source


def test_colab_is_secure_staged_chinese_and_has_live_copy_progress() -> None:
    notebook_path = (
        ROOT / "notebooks" / "YOLO11n_InceptionDW_SCSharedHead.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    all_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "getpass.getpass(" in code_source
    assert "credential.helper=" in code_source
    assert "ThreadPoolExecutor" in code_source
    assert "as_completed" in code_source
    assert "shutil.copyfile(" in code_source
    assert "tqdm(" in code_source
    assert 'desc="文件进度"' in code_source
    assert 'desc="字节进度"' in code_source
    assert "EXPECTED_COUNTS" not in code_source
    assert "total_epochs=150" in code_source
    assert "stage_epochs=80" in code_source
    assert "stage80_resume.pt" in all_source
    assert "CONFIRM_CONTINUE_TO_150 = False" in code_source
    assert "本单元格只展示数据，不自动替你作继续训练的决定" in code_source
    assert "实时显示日志" in code_source

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        compile_source = "\n".join(
            "pass  # 已单独检查 Colab 魔法命令"
            if line.lstrip().startswith("%")
            else line
            for line in "".join(cell.get("source", [])).splitlines()
        )
        compile(compile_source, notebook_path.name, "exec")
