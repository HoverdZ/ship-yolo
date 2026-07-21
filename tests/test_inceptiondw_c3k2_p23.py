"""Tests for the scoped YOLO11n InceptionDW P2/P3 experiment."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.c3k2_inceptiondw import InceptionDWBottleneck
from custom_modules.inceptiondw import InceptionDWConv2d
from tools.inceptiondw_utils import (
    MODEL_YAML,
    build_custom_model,
    full_check,
    structure_report,
    transfer_pretrained_weights,
    yaml_scope_report,
)
from tools.train_inceptiondw_c3k2_p23 import (
    METADATA_FILE,
    prepare_new_run_directory,
    validate_data_yaml,
)


class OfficialInceptionDWConv2dReference(nn.Module):
    """Minimal direct transcription of sail-sg/inceptionnext."""

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(
            gc,
            gc,
            square_kernel_size,
            padding=square_kernel_size // 2,
            groups=gc,
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (
                x_id,
                self.dwconv_hw(x_hw),
                self.dwconv_w(x_w),
                self.dwconv_h(x_h),
            ),
            dim=1,
        )


@pytest.mark.parametrize("channels", [16, 32, 64])
def test_inceptiondw_matches_official_reference(channels: int) -> None:
    torch.manual_seed(7)
    project = InceptionDWConv2d(channels).eval()
    reference = OfficialInceptionDWConv2dReference(channels).eval()
    reference.load_state_dict(project.state_dict(), strict=True)
    x = torch.randn(2, channels, 23, 19)
    with torch.inference_mode():
        project_output = project(x)
        reference_output = reference(x)
    assert project_output.shape == x.shape
    assert reference_output.shape == x.shape
    assert sum(project.split_indexes) == channels
    assert project.split_indexes == reference.split_indexes
    assert (project_output - reference_output).abs().max().item() <= 1e-7


def test_inceptiondw_branch_geometry_and_depthwise_groups() -> None:
    module = InceptionDWConv2d(64)
    assert module.split_indexes == (40, 8, 8, 8)
    assert module.dwconv_hw.kernel_size == (3, 3)
    assert module.dwconv_w.kernel_size == (1, 11)
    assert module.dwconv_h.kernel_size == (11, 1)
    assert module.dwconv_hw.groups == module.branch_channels
    assert module.dwconv_w.groups == module.branch_channels
    assert module.dwconv_h.groups == module.branch_channels


def test_bottleneck_preserves_cv1_shape_expansion_and_wrapper_order() -> None:
    bottleneck = InceptionDWBottleneck(16, 16, shortcut=True, e=0.5)
    assert bottleneck.cv1.conv.in_channels == 16
    assert bottleneck.cv1.conv.out_channels == 8
    assert bottleneck.cv1.conv.kernel_size == (3, 3)
    assert isinstance(bottleneck.cv2_adapter, nn.Conv2d)
    assert (bottleneck.cv2_adapter.in_channels, bottleneck.cv2_adapter.out_channels) == (8, 16)
    assert list(bottleneck.cv2._modules) == ["inception", "bn", "act"]
    assert bottleneck.add
    x = torch.randn(1, 16, 20, 20)
    assert bottleneck(x).shape == x.shape


def test_yaml_diff_is_limited_to_backbone_layers_2_and_4() -> None:
    report = yaml_scope_report(MODEL_YAML)
    assert report["all_checks_passed"], report


def test_dynamic_registration_builds_exact_scoped_structure() -> None:
    model = build_custom_model()
    report = structure_report(model)
    assert report["all_checks_passed"], report
    assert report["custom_c3k2_indices"] == [2, 4]
    assert report["official_c3k2_indices"] == [6, 8, 13, 16, 19, 22]
    assert report["inceptiondw_modules"] == [
        "model.2.m.0.cv2.inception",
        "model.4.m.0.cv2.inception",
    ]


def test_weight_transfer_inherits_untouched_scope_and_rejects_old_cv2() -> None:
    model = build_custom_model()
    report = transfer_pretrained_weights(model, ROOT / "yolo11n.pt", apply=True)
    assert all(report["p2_p3_cv1_inherited"].values())
    assert all(report["p2_p3_outer_1x1_inherited"].values())
    assert not report["untouched_backbone_unmatched"]
    assert not report["neck_unmatched"]
    assert not report["detect_unmatched"]
    assert not report["replaced_cv2_source_conv_keys_inherited"]
    assert report["inherited_tensors"] < report["target_state_tensors"]
    assert report["parameter_element_inheritance_ratio"] > 0.99


def test_full_640_cpu_non_training_check() -> None:
    report = full_check(weights=ROOT / "yolo11n.pt", imgsz=640)
    assert report["all_checks_passed"], report["checks"]
    assert report["forward"]["custom"]["detect_input_shapes"] == [
        [1, 64, 80, 80],
        [1, 128, 40, 40],
        [1, 256, 20, 20],
    ]
    assert report["forward"]["custom"]["all_outputs_finite"]


def test_train_script_is_direct_process_and_cli_help_works() -> None:
    import tools.train_inceptiondw_c3k2_p23 as train_script

    source = inspect.getsource(train_script)
    assert "subprocess" not in source
    result = subprocess.run(
        [sys.executable, "tools/train_inceptiondw_c3k2_p23.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--data" in result.stdout
    assert "--resume" in result.stdout


def test_training_guards_missing_data_and_existing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_data_yaml(tmp_path / "missing.yaml")

    run_dir = tmp_path / "run"
    prepare_new_run_directory(run_dir)
    (run_dir / METADATA_FILE).write_text("{}\n", encoding="utf-8")
    prepare_new_run_directory(run_dir)
    (run_dir / "results.csv").write_text("epoch\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_new_run_directory(run_dir)
