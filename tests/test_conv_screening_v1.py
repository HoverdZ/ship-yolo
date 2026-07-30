"""Tests for the controlled PConv/LSKConv/PKIConv screening experiments."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv

from custom_modules.c3k2_conv_screening import (
    C3k2_LSKConv,
    C3k2_PConv,
    C3k2_PKIConv,
)
from custom_modules.lsk_conv import LargeSelectiveKernelConv2d
from custom_modules.pinwheel_conv import PinwheelConv
from custom_modules.pki_conv import PolyKernelConv2d
from custom_modules.register import register_conv_screening_modules
from tools.conv_screening_utils import (
    ConvScreeningConfig,
    copy_dataset_to_local,
    cpu_forward_backward,
    install_trainer_handoff_guard,
    prepare_model,
    resolve_run_state,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "C1_pconv_p23": (
        ROOT / "experiments/conv_screening_v1/C1_yolo11n_pconv_p23.yaml",
        C3k2_PConv,
    ),
    "C2_lskconv_p23": (
        ROOT / "experiments/conv_screening_v1/C2_yolo11n_lskconv_p23.yaml",
        C3k2_LSKConv,
    ),
    "C3_pkiconv_p23": (
        ROOT / "experiments/conv_screening_v1/C3_yolo11n_pkiconv_p23.yaml",
        C3k2_PKIConv,
    ),
}


class OfficialPConvReference(nn.Module):
    """Literal module layout from the official PConv implementation."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1) -> None:
        super().__init__()
        paddings = ((k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k))
        self.pad = nn.ModuleList(nn.ZeroPad2d(padding) for padding in paddings)
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cat(
            torch.cat(
                (
                    self.cw(self.pad[0](x)),
                    self.cw(self.pad[1](x)),
                    self.ch(self.pad[2](x)),
                    self.ch(self.pad[3](x)),
                ),
                dim=1,
            )
        )


class OfficialLSKReference(nn.Module):
    """Literal LSKblock convolutional core from the official LSKNet repo."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            7,
            stride=1,
            padding=9,
            groups=dim,
            dilation=3,
        )
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_1 = self.conv0(x)
        attention_2 = self.conv_spatial(attention_1)
        attention_1 = self.conv1(attention_1)
        attention_2 = self.conv2(attention_2)
        attention = torch.cat((attention_1, attention_2), dim=1)
        average = torch.mean(attention, dim=1, keepdim=True)
        maximum = torch.max(attention, dim=1, keepdim=True).values
        selection = self.conv_squeeze(torch.cat((average, maximum), dim=1)).sigmoid()
        attention = attention_1 * selection[:, 0:1] + attention_2 * selection[:, 1:2]
        return x * self.conv(attention)


class OfficialPKIMixerReference(nn.Module):
    """PKINet dense non-dilated poly-kernel mixer without CAA/FFN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.base_conv = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.branch_convs = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel,
                padding=kernel // 2,
                groups=channels,
                bias=False,
            )
            for kernel in (5, 7, 9, 11)
        )
        self.project = Conv(channels, channels, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_conv(x)
        mixed = x
        for branch in self.branch_convs:
            mixed = mixed + branch(x)
        return self.project(mixed)


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        (PinwheelConv(8, 16), OfficialPConvReference(8, 16)),
        (LargeSelectiveKernelConv2d(16), OfficialLSKReference(16)),
        (PolyKernelConv2d(16), OfficialPKIMixerReference(16)),
    ],
)
def test_adapted_operator_matches_official_reference(
    candidate: nn.Module,
    reference: nn.Module,
) -> None:
    reference.load_state_dict(candidate.state_dict(), strict=True)
    candidate.eval()
    reference.eval()
    x = torch.randn(2, 16 if isinstance(candidate, (LargeSelectiveKernelConv2d, PolyKernelConv2d)) else 8, 31, 29)
    with torch.no_grad():
        actual = candidate(x)
        expected = reference(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape[-2:] == x.shape[-2:]


def test_registration_is_idempotent() -> None:
    register_conv_screening_modules()
    register_conv_screening_modules()


@pytest.mark.parametrize(("experiment_id", "entry"), EXPERIMENTS.items())
def test_model_scope_forward_backward_and_detect_contract(
    experiment_id: str,
    entry: tuple[Path, type[nn.Module]],
) -> None:
    yaml_path, expected_type = entry
    register_conv_screening_modules()
    wrapper = YOLO(str(yaml_path), verbose=False)
    layers = wrapper.model.model
    assert isinstance(layers[2], expected_type)
    assert isinstance(layers[4], expected_type)
    custom_types = (C3k2_PConv, C3k2_LSKConv, C3k2_PKIConv)
    assert [
        index for index, layer in enumerate(layers)
        if isinstance(layer, custom_types)
    ] == [2, 4]
    assert list(layers[-1].f) == [16, 19, 22]
    assert wrapper.model.stride.tolist() == [8.0, 16.0, 32.0]
    for index in (0, 1, 3, 5, 7):
        assert isinstance(layers[index], Conv)
        assert layers[index].conv.kernel_size == (3, 3)
        assert layers[index].conv.stride == (2, 2)
    for index in (2, 4):
        for bottleneck in layers[index].m:
            assert isinstance(bottleneck.cv1, Conv)
            assert bottleneck.cv1.conv.kernel_size == (3, 3)
            assert bottleneck.cv1.conv.stride == (1, 1)
            assert (
                bottleneck.cv1.conv.in_channels
                == 2 * bottleneck.cv1.conv.out_channels
            )
    forbidden = {
        "C3k2_InceptionDW",
        "DySample",
        "SCAM",
        "CASCAM",
        "VGUPPreprocessor",
        "ERUPPreprocessor",
    }
    assert not forbidden.intersection(
        type(module).__name__ for module in wrapper.model.modules()
    )
    assert cpu_forward_backward(wrapper, imgsz=64, seed=0)["passed"]


@pytest.mark.parametrize("experiment_id", EXPERIMENTS)
def test_official_transfer_preserves_first_conv_only_changes_cv2(
    tmp_path: Path,
    experiment_id: str,
) -> None:
    local_root = tmp_path / "local_data"
    config = ConvScreeningConfig(
        experiment_id=experiment_id,
        local_data_root=str(local_root),
        drive_runs_root=str(tmp_path / "runs"),
    )
    config.local_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(local_root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "ship"},
                "nc": 1,
            }
        ),
        encoding="utf-8",
    )
    prepared = prepare_model(
        config,
        ROOT / "yolo11n.pt",
        run_cpu_check=False,
    )
    transfer = prepared["transfer"]
    assert transfer["passed"]
    assert not transfer["p2_p3_cv1_missing_keys"]
    assert not transfer["out_of_scope_unmatched_target_keys"]
    assert transfer["loaded_tensors"] < transfer["target_state_tensors"]
    official = YOLO(str(ROOT / "yolo11n.pt"), verbose=False)
    target_state = prepared["model"].model.state_dict()
    source_state = official.model.float().state_dict()
    for key in transfer["p2_p3_cv1_expected_keys"]:
        torch.testing.assert_close(target_state[key], source_state[key], rtol=0, atol=0)


def test_dataset_copy_has_live_copy_contract_without_fixed_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "drive_data"
    destination = tmp_path / "local_data"
    (source / "images/train").mkdir(parents=True)
    (source / "images/val").mkdir(parents=True)
    (source / "labels/train").mkdir(parents=True)
    (source / "labels/val").mkdir(parents=True)
    (source / "images/train/a.jpg").write_bytes(b"train-image")
    (source / "images/val/b.jpg").write_bytes(b"val-image")
    (source / "labels/train/a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (source / "labels/val/b.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (source / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(source),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "ship"},
            }
        ),
        encoding="utf-8",
    )
    config = ConvScreeningConfig(
        experiment_id="C1_pconv_p23",
        drive_data_root=str(source),
        local_data_root=str(destination),
        drive_runs_root=str(tmp_path / "runs"),
        copy_workers=2,
    )
    report = copy_dataset_to_local(config)
    assert report["files_processed"] == 5
    assert report["fixed_count_comparison_performed"] is False
    assert report["splits_preserved"] == ["train", "val"]
    assert (destination / "images/train/a.jpg").read_bytes() == b"train-image"
    local_payload = yaml.safe_load(config.local_yaml.read_text(encoding="utf-8"))
    assert local_payload["path"] == str(destination)
    assert local_payload["nc"] == 1


def test_existing_residue_allocates_retry_without_deletion(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    residue = runs / "yolo11n_pconv_p23_640"
    residue.mkdir(parents=True)
    marker = residue / "partial.txt"
    marker.write_text("preserve me", encoding="utf-8")
    config = ConvScreeningConfig(
        experiment_id="C1_pconv_p23",
        drive_runs_root=str(runs),
    )
    state = resolve_run_state(config)
    assert state["mode"] == "new"
    assert state["run_name"] == "yolo11n_pconv_p23_640_retry1"
    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_trainer_handoff_guard_accepts_exact_state_and_rejects_replacement(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "local_data"
    config = ConvScreeningConfig(
        experiment_id="C1_pconv_p23",
        local_data_root=str(local_root),
        drive_runs_root=str(tmp_path / "runs"),
    )
    config.local_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(local_root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "ship"},
                "nc": 1,
            }
        ),
        encoding="utf-8",
    )
    prepared = prepare_model(config, ROOT / "yolo11n.pt")
    wrapper = prepared["model"]
    expected_run = tmp_path / "runs" / "exact"
    install_trainer_handoff_guard(
        wrapper,
        prepared["transfer"],
        expected_run / "preflight" / "trainer_handoff_report.json",
        expected_run_dir=expected_run,
    )

    class Trainer:
        save_dir = expected_run
        model = wrapper.model

    wrapper.callbacks["on_pretrain_routine_start"][-1](Trainer())
    wrapper.callbacks["on_pretrain_routine_end"][-1](Trainer())
    report = yaml.safe_load(
        (expected_run / "preflight" / "trainer_handoff_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["passed"]
    first_key = prepared["transfer"]["loaded_target_keys"][0]
    with torch.no_grad():
        wrapper.model.state_dict()[first_key].view(-1)[0].add_(1)
    with pytest.raises(RuntimeError, match="discarded or altered"):
        wrapper.callbacks["on_pretrain_routine_end"][-1](Trainer())
