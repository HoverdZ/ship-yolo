"""Unit tests for the shared ERUP BPW and KBL filters."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.erup import (
    BPWFilter,
    BPW_PARAMETER_COUNT,
    ERUPParameterEncoder,
    ERUPPreprocessor,
    ERUP_PARAMETER_COUNT,
    KBLFilter,
    KBL_PARAMETER_COUNT,
)


def test_bpw_zero_parameters_are_identity_for_batch_two() -> None:
    image = torch.rand(2, 3, 17, 19)
    params = torch.zeros(2, BPW_PARAMETER_COUNT)
    output = BPWFilter()(image, params)
    assert output.shape == image.shape
    assert torch.allclose(output, image, atol=2e-6, rtol=1e-5)
    assert output.amin() >= 0
    assert output.amax() <= 1


def test_bpw_channels_are_independent_and_gradients_flow() -> None:
    image = torch.full((1, 3, 9, 9), 0.4, requires_grad=True)
    params = torch.zeros(1, BPW_PARAMETER_COUNT, requires_grad=True)
    changed = params.detach().clone()
    changed[0, :4] = torch.tensor([0.8, -0.6, -0.4, 0.7])
    changed.requires_grad_(True)
    output = BPWFilter()(image, changed)
    assert not torch.allclose(output[:, 0], image[:, 0])
    assert torch.allclose(output[:, 1:], image[:, 1:], atol=2e-6)
    output.mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert changed.grad is not None and torch.isfinite(changed.grad).all()


def test_kbl_zero_parameters_are_identity_and_preserve_shape() -> None:
    image = torch.rand(2, 3, 11, 13)
    params = torch.zeros(2, KBL_PARAMETER_COUNT)
    output = KBLFilter()(image, params)
    assert output.shape == image.shape
    assert torch.equal(output, image)


def test_kbl_keeps_samples_and_rgb_channels_independent() -> None:
    image = torch.ones(2, 3, 9, 9)
    params = torch.zeros(2, KBL_PARAMETER_COUNT)
    kernels = params.view(2, 2, 3, 9, 9)
    kernels[1, 1, 2, 4, 4] = 0.5
    output = KBLFilter()(image, params)
    assert torch.equal(output[0], image[0])
    assert torch.equal(output[1, 0], image[1, 0])
    assert torch.equal(output[1, 1], image[1, 1])
    assert torch.allclose(output[1, 2], torch.full_like(output[1, 2], 1.5))


def test_kbl_gradients_reach_both_dynamic_kernels() -> None:
    image = torch.rand(2, 3, 9, 9, requires_grad=True)
    params = torch.zeros(2, KBL_PARAMETER_COUNT, requires_grad=True)
    output = KBLFilter()(image, params)
    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert params.grad is not None and torch.isfinite(params.grad).all()
    kernel_grads = params.grad.view(2, 2, 3, 9, 9)
    assert torch.count_nonzero(kernel_grads[:, 0]) > 0
    assert torch.count_nonzero(kernel_grads[:, 1]) > 0


def test_erup_uses_original_encoder_without_vgup_gates() -> None:
    module = ERUPPreprocessor()
    assert isinstance(module.encoder, ERUPParameterEncoder)
    assert module.parameter_count == ERUP_PARAMETER_COUNT == 498
    assert BPW_PARAMETER_COUNT == 12
    assert KBL_PARAMETER_COUNT == 486
    names = tuple(name for name, _ in module.named_modules())
    assert not any("gate" in name for name in names)


def test_erup_debug_split_and_backward() -> None:
    module = ERUPPreprocessor()
    image = torch.rand(1, 3, 64, 64, requires_grad=True)
    output, debug = module(image, return_debug=True)
    assert output.shape == image.shape
    assert debug["bpw_params"].shape == (1, 12)
    assert debug["kbl_params"].shape == (1, 486)
    assert debug["bpw_image"].shape == image.shape
    assert torch.isfinite(output).all()
    output.mean().backward()
    assert module.encoder.projection.weight.grad is not None
    assert torch.isfinite(module.encoder.projection.weight.grad).all()
