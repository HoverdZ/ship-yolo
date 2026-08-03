"""Checks for the YOLOv8-native scoped InceptionDW adaptation."""

from __future__ import annotations

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv

from custom_modules.c2f_inceptiondw import C2f_InceptionDW
from custom_modules.c3k2_inceptiondw import InceptionDWBottleneck
from custom_modules.inceptiondw import InceptionDWConvBNAct
from custom_modules.register import register_custom_modules
from tools.formal_experiments.registry import ROOT


def _flatten(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _flatten(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten(child)


def test_c2f_inceptiondw_preserves_native_first_spatial_conv() -> None:
    target = C2f_InceptionDW(64, 64, n=2, shortcut=True)
    reference = C2f(64, 64, n=2, shortcut=True)

    assert target.cv1.conv.weight.shape == reference.cv1.conv.weight.shape
    assert target.cv2.conv.weight.shape == reference.cv2.conv.weight.shape
    for adapted, official in zip(target.m, reference.m, strict=True):
        assert isinstance(adapted, InceptionDWBottleneck)
        assert isinstance(adapted.cv1, Conv)
        assert adapted.cv1.conv.weight.shape == official.cv1.conv.weight.shape
        assert isinstance(adapted.cv2_adapter, nn.Identity)
        assert isinstance(adapted.cv2, InceptionDWConvBNAct)


def test_r12_uses_inceptiondw_only_in_p2_p3_c2f() -> None:
    register_custom_modules()
    path = (
        ROOT
        / "experiments/formal_models/"
        "R12_yolov8n_inceptiondw_dpls_ca_scam_vgup.yaml"
    )
    network = YOLO(str(path), verbose=False).model

    assert isinstance(network.model[3], C2f_InceptionDW)
    assert isinstance(network.model[5], C2f_InceptionDW)
    assert isinstance(network.model[7], C2f)
    assert not isinstance(network.model[7], C2f_InceptionDW)

    network.eval()
    image = torch.randn(1, 3, 64, 64, requires_grad=True)
    output = network(image)
    tensors = list(_flatten(output))
    assert tensors
    loss = sum(value.float().square().mean() for value in tensors)
    loss.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
