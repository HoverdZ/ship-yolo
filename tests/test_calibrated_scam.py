"""Unit tests for equivalent-initialized Contrast-Aware SCAM."""

from __future__ import annotations

import torch

from custom_modules.calibrated_scam import CASCAM
from custom_modules.scam import SCAM


def _legacy_forward(module: SCAM, x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    avg = module.avg_pool(x).softmax(dim=1).view(batch, 1, 1, channels)
    maximum = module.max_pool(x).softmax(dim=1).view(batch, 1, 1, channels)
    k = module.k(x).view(batch, 1, -1, 1).softmax(dim=2)
    v = module.v(x).view(batch, 1, channels, -1)
    channel_context = torch.matmul(v, k).view(batch, channels, 1, 1)
    spatial_context = torch.cat(
        (
            torch.matmul(avg, v).view(batch, 1, height, width),
            torch.matmul(maximum, v).view(batch, 1, height, width),
        ),
        dim=1,
    )
    return x + module.m(channel_context) * module.m2(spatial_context).sigmoid()


def test_scam_refactor_is_bit_exact_and_keeps_state_keys() -> None:
    torch.manual_seed(1)
    module = SCAM(16).eval()
    state_keys = set(module.state_dict())
    image = torch.randn(2, 16, 7, 9)
    with torch.inference_mode():
        assert torch.equal(module(image), _legacy_forward(module, image))
    assert set(module.state_dict()) == state_keys
    assert all(
        not key.startswith(("local_mean", "contrast_proj", "contrast_logit"))
        for key in state_keys
    )


def test_ca_scam_initial_output_is_exact_original_scam() -> None:
    torch.manual_seed(2)
    original = SCAM(16).eval()
    calibrated = CASCAM(16, max_delta=0.1).eval()
    load = calibrated.load_state_dict(original.state_dict(), strict=False)
    assert sorted(load.missing_keys) == [
        "contrast_logit",
        "contrast_proj.bias",
        "contrast_proj.weight",
    ]
    assert not load.unexpected_keys
    image = torch.randn(2, 16, 8, 10)
    with torch.inference_mode():
        assert torch.equal(calibrated(image), original(image))


def test_local_contrast_constant_and_edge_inputs() -> None:
    module = CASCAM(8)
    constant = torch.ones(1, 8, 9, 9)
    local, contrast_map, beta = module.contrast_state(constant)
    assert torch.count_nonzero(local) == 0
    assert torch.equal(contrast_map, torch.full_like(contrast_map, 0.5))
    assert beta.item() == 0.0

    edge = torch.zeros(1, 8, 9, 9)
    edge[:, :, :, 4:] = 1
    with torch.no_grad():
        module.contrast_proj.weight.zero_()
        module.contrast_proj.weight[0, 0, 1, 1] = 1.0
    local, contrast_map, _ = module.contrast_state(edge)
    assert local.max() > local.min()
    assert contrast_map.max() > contrast_map.min()


def test_beta_is_bounded() -> None:
    module = CASCAM(8, max_delta=0.1)
    with torch.no_grad():
        module.contrast_logit.fill_(100)
    _, _, positive = module.contrast_state(torch.randn(1, 8, 5, 5))
    with torch.no_grad():
        module.contrast_logit.fill_(-100)
    _, _, negative = module.contrast_state(torch.randn(1, 8, 5, 5))
    assert 0 < positive.item() <= 0.100001
    assert -0.100001 <= negative.item() < 0


def test_gradients_reach_logit_then_projection_after_gate_opens() -> None:
    module = CASCAM(8)
    image = torch.randn(2, 8, 7, 7, requires_grad=True)
    module(image).square().mean().backward()
    assert module.contrast_logit.grad is not None
    assert torch.isfinite(module.contrast_logit.grad).all()

    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.contrast_logit.fill_(0.2)
    module(image.detach()).square().mean().backward()
    assert module.contrast_proj.weight.grad is not None
    assert torch.count_nonzero(module.contrast_proj.weight.grad) > 0
    assert torch.isfinite(module.contrast_proj.weight.grad).all()


def test_ca_scam_roundtrip_and_independent_instances() -> None:
    modules = [CASCAM(8) for _ in range(3)]
    parameter_ids = [
        {id(parameter) for parameter in module.parameters()}
        for module in modules
    ]
    assert all(
        parameter_ids[left].isdisjoint(parameter_ids[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    clone = CASCAM(8)
    result = clone.load_state_dict(modules[0].state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
