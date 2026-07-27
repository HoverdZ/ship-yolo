"""Build and topology tests for cumulative DySample/PLS/SCAM models."""

from __future__ import annotations

import pytest
import torch

from custom_modules.register import register_cumulative_modules
from tools.cumulative_models_utils import (
    EXPERIMENTS,
    build_model,
    compatibility_report,
    forward_report,
    model_statistics,
    structure_report,
)


def test_colab_training_is_not_launched_in_a_subprocess() -> None:
    from tools.build_cumulative_notebook import build_notebook

    notebook = build_notebook()
    training_cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "RUN_TRAINING" in cell.source
    ]
    assert len(training_cells) == 1
    source = training_cells[0]
    assert "train_experiment(" in source
    assert "RUN_TRAINING = True" in source
    assert 'cache="disk"' in source
    assert "deterministic=False" in source
    assert "subprocess.run" not in source
    assert "!python" not in source
    assert "%run" not in source


def test_registration_is_idempotent_and_exposes_custom_modules() -> None:
    register_cumulative_modules()
    first_parser = None
    import ultralytics.nn.tasks as tasks

    first_parser = tasks.parse_model
    register_cumulative_modules()

    from custom_modules.dysample import DySample
    from custom_modules.scam import SCAM

    assert tasks.parse_model is first_parser
    assert tasks.DySample is DySample
    assert tasks.SCAM is SCAM
    assert tasks.parse_model._ship_yolo_patch_version == 6


@pytest.mark.parametrize("experiment", tuple(EXPERIMENTS))
def test_cumulative_model_build_stride_and_forward(
    experiment: str,
) -> None:
    model = build_model(experiment)
    structure = structure_report(model, experiment)
    stats = model_statistics(model, imgsz=64)
    forward = forward_report(model, experiment, imgsz=64)
    assert structure["passed"], structure
    assert stats["parameters"] > 0
    assert stats["gflops"] > 0
    assert forward["passed"], forward


@pytest.mark.parametrize(
    "experiment",
    ("incdw_dysample_pls", "incdw_dysample_pls_scam"),
)
def test_pls_models_have_no_p5_detection_or_backbone_stage(
    experiment: str,
) -> None:
    model = build_model(experiment)
    backbone = model.model.yaml["backbone"]
    assert len(backbone) == 9
    assert [float(value) for value in model.model.stride] == [
        4.0,
        8.0,
        16.0,
    ]
    assert 32.0 not in [
        float(value) for value in model.model.stride
    ]
    assert model.model.model[-1].f == EXPERIMENTS[experiment][
        "detect_from"
    ]


def test_scam_model_uses_three_independent_instances() -> None:
    from custom_modules.scam import SCAM

    model = build_model("incdw_dysample_pls_scam")
    modules = [
        item
        for item in model.model.modules()
        if isinstance(item, SCAM)
    ]
    assert len(modules) == 3
    ids = [
        {id(parameter) for parameter in module.parameters()}
        for module in modules
    ]
    assert all(
        left.isdisjoint(right)
        for offset, left in enumerate(ids)
        for right in ids[offset + 1 :]
    )


def test_native_and_existing_inception_models_still_build() -> None:
    report = compatibility_report()
    assert report["passed"], report


def test_cpu_backward_on_largest_cumulative_model() -> None:
    model = build_model("incdw_dysample_pls_scam")
    network = model.model.cpu().train()
    image = torch.randn(1, 3, 64, 64, requires_grad=True)
    output = network(image)

    def tensors(value):
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, dict):
            return [
                tensor
                for item in value.values()
                for tensor in tensors(item)
            ]
        if isinstance(value, (list, tuple)):
            return [
                tensor
                for item in value
                for tensor in tensors(item)
            ]
        return []

    loss = sum(item.float().square().mean() for item in tensors(output))
    loss.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
