"""Pre-training tests for the BADC, SCG, and SGTA experiment family."""

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

from custom_modules.badc import BackgroundAwareDirectionalContrast
from custom_modules.scg import SemanticConfirmationGate
from custom_modules.sgta import (
    ScaleAdaptiveGaussianBboxLoss,
    iou_blend_weight,
    normalized_wasserstein_similarity,
    scale_adaptive_quality,
)
from tools.badc_scg_sgta_utils import (
    VARIANTS,
    build_model,
    forward_report,
    structure_report,
    transfer_weights,
)
from tools.train_badc_scg_sgta import (
    STAGE_CHECKPOINT,
    _attach_stage_callbacks,
    validate_resume_checkpoint,
)


def test_badc_constant_background_has_zero_contrast() -> None:
    module = BackgroundAwareDirectionalContrast(8)
    x = torch.full((2, 8, 16, 16), 3.0)
    background, contrast = module.decompose(x)
    assert torch.allclose(background, x)
    assert torch.count_nonzero(contrast) == 0
    weights = module.branch_weights(background, contrast)
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0))
    assert torch.count_nonzero(module(x)) == 0


def test_scg_is_native_concat_at_initialization_and_bounded() -> None:
    module = SemanticConfirmationGate([16, 32], hidden_ratio=0.25, alpha_max=0.25)
    module.eval()
    c3 = torch.randn(2, 16, 20, 20)
    semantic = torch.randn(2, 32, 20, 20)
    with torch.inference_mode():
        modulation = module.modulation(c3, semantic)
        output = module([c3, semantic])
    assert torch.equal(modulation, torch.ones_like(modulation))
    assert torch.equal(output, torch.cat((semantic, c3), dim=1))
    assert output.shape == (2, 48, 20, 20)


def test_nwd_is_one_for_identical_boxes_and_symmetric() -> None:
    boxes = torch.tensor([[0.0, 0.0, 8.0, 4.0], [10.0, 20.0, 50.0, 70.0]])
    assert torch.allclose(
        normalized_wasserstein_similarity(boxes, boxes), torch.ones(2), atol=1e-5
    )
    shifted = boxes + torch.tensor([1.0, 0.0, 1.0, 0.0])
    forward = normalized_wasserstein_similarity(boxes, shifted)
    backward = normalized_wasserstein_similarity(shifted, boxes)
    assert torch.allclose(forward, backward)
    assert bool(((forward > 0) & (forward < 1)).all())


def test_scale_adaptive_quality_prefers_nwd_for_tiny_and_iou_for_large() -> None:
    tiny_gt = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    tiny_pred = torch.tensor([[1.0, 0.0, 5.0, 4.0]])
    tiny_quality, tiny_iou, tiny_nwd = scale_adaptive_quality(tiny_gt, tiny_pred)
    assert tiny_nwd > tiny_iou
    assert tiny_quality > tiny_iou
    assert iou_blend_weight(tiny_gt) < 0.02

    large_gt = torch.tensor([[0.0, 0.0, 96.0, 96.0]])
    large_pred = torch.tensor([[2.0, 0.0, 98.0, 96.0]])
    large_quality, large_iou, _ = scale_adaptive_quality(large_gt, large_pred)
    assert iou_blend_weight(large_gt) > 0.99
    assert torch.allclose(large_quality, large_iou, atol=1e-4)


def test_sgta_bbox_loss_is_finite_and_backpropagates() -> None:
    loss_module = ScaleAdaptiveGaussianBboxLoss(reg_max=16)
    pred_dist = torch.randn(1, 4, 64, requires_grad=True)
    pred_bboxes = torch.tensor(
        [[[0.0, 0.0, 1.1, 1.0], [1.0, 1.0, 2.0, 2.0],
          [2.0, 2.0, 3.0, 3.0], [3.0, 3.0, 4.0, 4.0]]],
        requires_grad=True,
    )
    anchor_points = torch.tensor(
        [[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]
    )
    target_bboxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0],
          [2.0, 2.0, 3.0, 3.0], [3.0, 3.0, 4.0, 4.0]]]
    )
    target_scores = torch.ones(1, 4, 1)
    fg_mask = torch.ones(1, 4, dtype=torch.bool)
    stride = torch.full((4, 1), 8.0)
    box_loss, dfl_loss = loss_module(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores.sum(),
        fg_mask,
        torch.tensor([640.0, 640.0]),
        stride,
    )
    total = box_loss + dfl_loss
    total.backward()
    assert torch.isfinite(total)
    assert pred_dist.grad is not None and torch.isfinite(pred_dist.grad).all()
    assert pred_bboxes.grad is not None and torch.isfinite(pred_bboxes.grad).all()


def test_sgta_detection_model_criterion_runs_end_to_end() -> None:
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

    from custom_modules.register import register_custom_modules
    from custom_modules.sgta import SGTADetectionModel

    register_custom_modules()
    model = SGTADetectionModel(
        ROOT / "experiments" / "yolo11n_inceptiondw_sgta.yaml",
        nc=1,
        ch=3,
        verbose=False,
    ).cpu().train()
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    batch = {
        "img": torch.randn(2, 3, 128, 128),
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.zeros((2, 1)),
        "bboxes": torch.tensor(
            [[0.50, 0.50, 0.05, 0.04], [0.30, 0.40, 0.10, 0.08]]
        ),
    }
    loss, items = model(batch)
    loss.sum().backward()
    first_gradient = model.model[0].conv.weight.grad
    assert list(items.shape) == [3]
    assert torch.isfinite(loss).all()
    assert first_gradient is not None and torch.isfinite(first_gradient).all()


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_model_structure_forward_and_weight_transfer(variant: str) -> None:
    model = build_model(variant)
    assert structure_report(variant, model)["all_checks_passed"]
    assert forward_report(variant, imgsz=128)["all_checks_passed"]
    transfer = transfer_weights(model, variant, ROOT / "yolo11n.pt")
    assert transfer["loaded_state_tensors"] >= (487 if variant in {"badc", "full"} else 497)
    assert transfer["loaded_target_parameter_element_ratio"] > 0.95


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


def test_resume_validation_requires_optimizer_and_150_epoch_horizon(tmp_path: Path) -> None:
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
    with pytest.raises(RuntimeError, match="not resumable"):
        validate_resume_checkpoint(invalid, 150)


def test_training_entry_is_direct_process() -> None:
    import tools.train_badc_scg_sgta as training

    source = inspect.getsource(training.run_training)
    assert "subprocess" not in source
    assert "model.train(" in source


def test_colab_is_secure_staged_and_has_live_copy_progress() -> None:
    notebook_path = (
        ROOT / "notebooks" / "YOLO11n_BADC_SCG_SGTA_Screening.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "getpass.getpass(" in source
    assert "credential.helper=" in source
    assert "ThreadPoolExecutor" in source
    assert "as_completed" in source
    assert "shutil.copyfile(" in source
    assert "tqdm(" in source
    assert 'desc="Files"' in source
    assert 'desc="Bytes"' in source
    assert "EXPECTED_COUNTS" not in source
    assert "total_epochs=150" in source
    assert "stage_epochs=80" in source
    assert "stage80_resume.pt" in source
    assert "CONFIRM_CONTINUE_TO_150 = False" in source
    assert "run_training(stage_request)" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        compile_source = "\n".join(
            "pass  # Colab magic validated separately"
            if line.lstrip().startswith("%")
            else line
            for line in "".join(cell.get("source", [])).splitlines()
        )
        compile(compile_source, notebook_path.name, "exec")
