from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from custom_modules.register import register_custom_modules
from custom_modules.ac_yolo_official import ACmix, C2PSA_ACmix
from custom_modules.remote_ship_reproductions import (
    C2fRFA,
    C2fRepGhost,
    DATBlock,
    FASFF,
    ShuffleAttention,
    WeightedFeatureFusion,
)
from tools.build_remote_ship_comparison_notebooks import build_notebook
from tools.external_baselines.ship_losses import (
    InnerMPDIoUDetectionModel,
    MPDIoUDetectionModel,
    WiseIoUv3DetectionModel,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "P02_yolov8n_ship_yolo.yaml": (8, 16, 32),
    "P03_yolov8n_pmf_yolov8.yaml": (4, 8, 16, 32),
    "P04_yolov8s_ewff_net.yaml": (8, 16, 32),
    "P05_yolo11n_ac_yolo.yaml": (8, 16, 32),
}


@pytest.fixture(scope="module", autouse=True)
def _register_modules():
    register_custom_modules()


@pytest.mark.parametrize(("filename", "expected_strides"), CONFIGS.items())
def test_paper_model_build_and_forward(filename: str, expected_strides: tuple[int, ...]):
    from ultralytics import YOLO

    path = ROOT / "experiments" / "paper_comparisons" / filename
    model = YOLO(str(path), task="detect").model.train()
    assert tuple(int(value) for value in model.stride.tolist()) == expected_strides
    sample = torch.randn(1, 3, 128, 128, requires_grad=True)
    output = model(sample)
    assert isinstance(output, dict)
    assert len(output["feats"]) == len(expected_strides)
    output["boxes"].square().mean().backward()
    assert sample.grad is not None
    assert torch.isfinite(sample.grad).all()


def test_each_reproduction_contains_its_published_modules():
    from ultralytics import YOLO

    root = ROOT / "experiments" / "paper_comparisons"
    models = {
        name: YOLO(str(root / name), task="detect").model
        for name in CONFIGS
    }
    assert any(isinstance(module, C2fRepGhost) for module in models["P02_yolov8n_ship_yolo.yaml"].modules())
    assert sum(isinstance(module, ShuffleAttention) for module in models["P02_yolov8n_ship_yolo.yaml"].modules()) == 2

    assert any(isinstance(module, C2fRFA) for module in models["P03_yolov8n_pmf_yolov8.yaml"].modules())
    assert sum(isinstance(module, FASFF) for module in models["P03_yolov8n_pmf_yolov8.yaml"].modules()) == 4

    assert any(isinstance(module, DATBlock) for module in models["P04_yolov8s_ewff_net.yaml"].modules())
    assert sum(isinstance(module, WeightedFeatureFusion) for module in models["P04_yolov8s_ewff_net.yaml"].modules()) == 5

    ac_yolo = models["P05_yolo11n_ac_yolo.yaml"]
    assert sum(isinstance(module, C2PSA_ACmix) for module in ac_yolo.modules()) == 1
    assert any(isinstance(module, ACmix) for module in ac_yolo.modules())


@pytest.mark.parametrize(
    ("model_class", "filename"),
    [
        (WiseIoUv3DetectionModel, "P02_yolov8n_ship_yolo.yaml"),
        (InnerMPDIoUDetectionModel, "P03_yolov8n_pmf_yolov8.yaml"),
        (MPDIoUDetectionModel, "P05_yolo11n_ac_yolo.yaml"),
    ],
)
def test_custom_bbox_criteria_are_finite(model_class, filename: str):
    path = ROOT / "experiments" / "paper_comparisons" / filename
    model = model_class(str(path), nc=1, verbose=False).train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    sample = torch.randn(1, 3, 128, 128, requires_grad=True)
    predictions = model(sample)
    batch = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1]]),
    }
    total, components = model.init_criterion()(predictions, batch)
    assert torch.isfinite(total).all()
    assert torch.isfinite(components).all()
    total.sum().backward()
    assert sample.grad is not None


def test_protocol_controls_and_notebooks_are_foreground_and_token_free():
    protocol = yaml.safe_load(
        (ROOT / "experiments" / "paper_comparisons" / "protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    training = protocol["training"]
    assert training["imgsz"] == 640
    assert training["epochs"] == 150
    assert training["batch"] == 8
    assert training["seed"] == 0
    assert training["cache"] == "disk"
    assert set(protocol["runs"]) == {"P02", "P03", "P04", "P05"}

    for run_id, item in protocol["runs"].items():
        notebook = build_notebook(run_id, item["method"], "a" * 40)
        text = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "GITHUB_TOKEN" not in text
        assert "RUN_TRAINING" not in text
        assert "train_foreground(run, data_yaml)" in text
        assert "source_root=\"/content/drive/MyDrive/ship_detection/data\"" in text
        assert "augment=False" in text
