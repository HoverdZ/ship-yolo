"""CPU checks for the Relative-Uncertainty Quality head."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace

from custom_modules.register import register_custom_modules
from custom_modules.ruq_head import RUQDetect, RUQDetectionLoss


ROOT = Path(__file__).resolve().parents[1]
BASELINE_YAML = ROOT / "experiments" / "model_ablation" / "A0_yolo11n.yaml"
RUQ_YAML = ROOT / "experiments" / "component_ablation" / "Q1_yolo11n_ruq_head.yaml"


def _model_with_train_args() -> YOLO:
    register_custom_modules()
    wrapper = YOLO(str(RUQ_YAML))
    wrapper.model.args = IterableSimpleNamespace(**wrapper.model.args)
    return wrapper


def _gaussian_logits(mean: float, sigma: float, reg_max: int = 16) -> torch.Tensor:
    bins = torch.arange(reg_max, dtype=torch.float32)
    return -0.5 * ((bins - mean) / sigma).square()


def test_relative_uncertainty_is_larger_for_narrower_prediction() -> None:
    head = RUQDetect(nc=1, ch=(64, 128, 256))
    small = torch.stack([_gaussian_logits(2.0, 0.7) for _ in range(4)]).reshape(1, 64, 1)
    large = torch.stack([_gaussian_logits(6.0, 0.7) for _ in range(4)]).reshape(1, 64, 1)

    small_stats = head.distribution_statistics(small)
    large_stats = head.distribution_statistics(large)

    assert small_stats.shape == (1, 10, 1)
    assert small_stats[0, 8, 0] > large_stats[0, 8, 0]
    assert small_stats[0, 9, 0] > large_stats[0, 9, 0]


def test_yaml_builds_ruq_head_and_routes_custom_loss() -> None:
    wrapper = _model_with_train_args()
    head = wrapper.model.model[-1]

    assert isinstance(head, RUQDetect)
    assert head.stride.tolist() == [8.0, 16.0, 32.0]
    assert isinstance(wrapper.model.init_criterion(), RUQDetectionLoss)


def test_native_yolo11n_state_is_fully_compatible() -> None:
    register_custom_modules()
    baseline = DetectionModel(str(BASELINE_YAML), ch=3, nc=80, verbose=False)
    ruq = DetectionModel(str(RUQ_YAML), ch=3, nc=80, verbose=False)

    baseline_state = baseline.state_dict()
    ruq_state = ruq.state_dict()
    assert all(key in ruq_state for key in baseline_state)
    assert all(ruq_state[key].shape == value.shape for key, value in baseline_state.items())

    incompatible = ruq.load_state_dict(baseline_state, strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "model.23.quality_predictor.0.weight",
        "model.23.quality_predictor.0.bias",
        "model.23.quality_predictor.2.weight",
        "model.23.quality_predictor.2.bias",
    }
    assert sum(parameter.numel() for parameter in ruq.parameters()) - sum(
        parameter.numel() for parameter in baseline.parameters()
    ) == 193


def test_cpu_forward_backward_and_score_calibration() -> None:
    wrapper = _model_with_train_args()
    model = wrapper.model
    head = model.model[-1]
    model.train()

    images = torch.randn(1, 3, 128, 128)
    predictions = model(images)
    batch = {
        "img": images,
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    loss, loss_items = model.loss(batch, predictions)
    loss.sum().backward()

    assert loss_items.shape == (3,)
    assert torch.isfinite(loss_items).all()
    assert head.quality_predictor[-1].weight.grad is not None
    assert head.quality_predictor[-1].weight.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        decoded, raw = model(torch.randn(1, 3, 128, 128))
    expected_scores = raw["scores"].sigmoid() * raw["quality"].sigmoid()
    torch.testing.assert_close(decoded[:, 4:], expected_scores)


def test_registration_is_idempotent() -> None:
    register_custom_modules()
    register_custom_modules()
