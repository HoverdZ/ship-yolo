"""Structural, numerical, transfer, and initialization tests for Prefusion."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torchvision.ops import DeformConv2d
from ultralytics.nn.modules import C3k2, Concat, Detect

from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
from custom_modules.fapn_prefusion import (
    FaPNAlignmentOnly,
    FaPNDepthwiseModulatedDeformConv2d,
    FaPNFeatureSelectionKeep,
)
from tools.fapn_prefusion_utils import (
    ALIGN_INDICES,
    CRITICAL_TENSOR_KEYS,
    FSM_INDICES,
    VARIANTS,
    backward_report,
    build_model,
    compare_parameter_shapes_with_official,
    forward_report,
    install_safe_prefusion_flops,
    load_init_model,
    prepare_formal_run_directory,
    semantic_weight_transfer,
    structure_report,
    topology_report,
    validate_init_manifest,
    variant_config,
    verify_prefusion_trainer_initialization,
)


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "yolo11n.pt"


@pytest.fixture(scope="session", params=("baseline", "inceptiondw"))
def variant_model(request):
    return request.param, build_model(request.param)


def test_both_yamls_build_and_topology_matches_contract() -> None:
    report = topology_report()
    assert report["all_checks_passed"], report


def test_top_level_types_and_from_relations(variant_model) -> None:
    variant, yolo = variant_model
    top = yolo.model.model
    expected_from = {
        11: -1,
        12: 6,
        13: [12, 11],
        14: [12, 13],
        15: -1,
        16: -1,
        17: 4,
        18: [17, 16],
        19: [17, 18],
        20: -1,
        21: -1,
        22: [-1, 15],
        23: -1,
        24: -1,
        25: [-1, 10],
        26: -1,
        27: [20, 23, 26],
    }
    assert len(top) == 28
    assert all(top[index].f == value for index, value in expected_from.items())
    assert all(isinstance(top[index], nn.Upsample) for index in (11, 16))
    assert all(top[index].mode == "nearest" for index in (11, 16))
    assert all(isinstance(top[index], Concat) for index in (14, 19, 22, 25))
    assert all(isinstance(top[index], C3k2) for index in (15, 20, 23, 26))
    assert isinstance(top[27], Detect)
    assert list(top[27].f) == [20, 23, 26]
    custom_indices = [
        index for index, layer in enumerate(top) if isinstance(layer, C3k2_InceptionDW)
    ]
    assert custom_indices == ([2, 4] if variant == "inceptiondw" else [])


def test_structure_report_covers_channels_groups_and_initialization(variant_model) -> None:
    variant, yolo = variant_model
    report = structure_report(yolo, variant)
    assert report["all_checks_passed"], report


def test_fsm_preserves_shape_channels_dtype_and_has_only_requested_ops() -> None:
    module = FaPNFeatureSelectionKeep(24).eval()
    x = torch.randn(2, 24, 9, 11, dtype=torch.float32)
    with torch.inference_mode():
        output = module(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert module.conv_attention.in_channels == module.conv_attention.out_channels == 24
    assert module.conv_attention.kernel_size == (1, 1)
    assert float(module.gamma_s.detach()) == pytest.approx(0.1)
    assert not any(isinstance(item, (nn.BatchNorm2d, nn.GroupNorm, nn.Dropout, nn.Softmax)) for item in module.modules())


@pytest.mark.parametrize("channels", (128, 256))
def test_depthwise_dcn_identity_and_offset_mask_rules(channels: int) -> None:
    module = FaPNDepthwiseModulatedDeformConv2d(channels).eval()
    high = torch.randn(2, channels, 13, 15)
    controller = torch.randn(2, 64, 13, 15)
    with torch.inference_mode():
        offset, mask = module.offset_and_mask(controller)
        output = module(high, controller)
    assert isinstance(module.dcn, DeformConv2d)
    assert module.dcn.groups == channels
    assert module.deformable_groups == 8
    assert offset.shape == (2, 144, 13, 15)
    assert mask.shape == (2, 72, 13, 15)
    assert torch.count_nonzero(module.conv_offset_mask.weight) == 0
    assert torch.count_nonzero(module.conv_offset_mask.bias) == 0
    assert torch.all(mask == 0.5)
    assert torch.max(torch.abs(output - high)).item() < 1e-5


@pytest.mark.parametrize("low_channels,high_channels", ((128, 256), (128, 128)))
def test_alignment_only_initial_residual_is_identity(low_channels: int, high_channels: int) -> None:
    module = FaPNAlignmentOnly(low_channels, high_channels).eval()
    low = torch.randn(1, low_channels, 10, 12)
    high = torch.randn(1, high_channels, 10, 12)
    with torch.inference_mode():
        output = module([low, high])
    assert output.shape == high.shape
    assert output.dtype == high.dtype
    assert module.out_channels == high_channels
    assert module.controller_channels == 64
    assert float(module.gamma_a.detach()) == pytest.approx(0.1)
    assert torch.max(torch.abs(output - high)).item() < 1e-5
    assert not any(isinstance(item, (nn.Upsample, nn.ReLU, nn.BatchNorm2d, nn.GroupNorm)) for item in module.modules())


def test_alignment_only_rejects_wrong_input_order_or_spatial_size() -> None:
    module = FaPNAlignmentOnly(128, 256)
    low = torch.randn(1, 128, 10, 10)
    high = torch.randn(1, 256, 5, 5)
    with pytest.raises(ValueError, match="does not upsample internally"):
        module([low, high])
    with pytest.raises(ValueError, match="expects"):
        module([low])


def test_forward_640_records_exact_channels_and_three_strides(variant_model) -> None:
    _variant, yolo = variant_model
    report = forward_report(yolo, imgsz=640)
    assert report["all_checks_passed"], report
    assert report["top_level_shapes"]["15"] == [1, 128, 40, 40]
    assert report["top_level_shapes"]["20"] == [1, 64, 80, 80]
    assert report["top_level_shapes"]["23"] == [1, 128, 40, 40]
    assert report["top_level_shapes"]["26"] == [1, 256, 20, 20]
    assert list(map(float, yolo.model.stride)) == [8.0, 16.0, 32.0]


def test_backward_256_gives_every_new_parameter_a_finite_gradient(variant_model) -> None:
    _variant, yolo = variant_model
    report = backward_report(yolo, imgsz=256)
    assert report["all_checks_passed"], report
    assert not report["missing_gradient_keys"]
    assert not report["nonfinite_gradient_keys"]


def test_unchanged_top_down_and_pan_parameter_shapes_match_official(variant_model) -> None:
    _variant, yolo = variant_model
    report = compare_parameter_shapes_with_official(yolo)
    assert report["all_checks_passed"], report


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="local official yolo11n.pt is required")
def test_semantic_weight_transfer_is_strict_and_never_loads_prefusion(variant_model) -> None:
    variant, yolo = variant_model
    report = semantic_weight_transfer(yolo, WEIGHTS, variant=variant, apply=True)
    assert report["all_strict_checks_passed"], report["strict_checks"]
    assert report["breakdown"]["top_down_c3k2_first"]["parameter_element_ratio"] == 1.0
    assert report["breakdown"]["top_down_c3k2_second"]["parameter_element_ratio"] == 1.0
    assert report["breakdown"]["pan"]["parameter_element_ratio"] == 1.0
    assert report["breakdown"]["prefusion_new"]["inherited_parameter_tensors"] == 0
    assert report["new_prefusion_random_parameter_keys"]
    if variant == "inceptiondw":
        assert report["inceptiondw_random_parameter_keys"]


@pytest.mark.parametrize("variant", ("baseline", "inceptiondw"))
def test_init_checkpoint_reloads_and_manifest_hashes_match(variant: str) -> None:
    config = variant_config(variant)
    init_pt = Path(config["init_pt"])
    manifest = Path(config["manifest"])
    if not init_pt.is_file():
        pytest.skip("ignored local initialization checkpoint has not been generated")
    report = validate_init_manifest(init_pt, manifest)
    assert report["all_checks_passed"], report
    loaded = load_init_model(init_pt)
    assert len(loaded.model.model) == 28
    assert list(loaded.model.model[27].f) == [20, 23, 26]
    assert all(key in loaded.model.state_dict() for key in CRITICAL_TENSOR_KEYS.values())


def test_trainer_callback_passes_then_detects_deliberate_weight_damage() -> None:
    config = variant_config("baseline")
    init_pt = Path(config["init_pt"])
    manifest = Path(config["manifest"])
    if not init_pt.is_file():
        pytest.skip("ignored local initialization checkpoint has not been generated")
    loaded = load_init_model(init_pt)
    from ultralytics.nn.tasks import DetectionModel

    # Reproduce DetectionTrainer.get_model(cfg=..., weights=...) rather than
    # checking only the checkpoint's directly unpickled model.
    rebuilt = DetectionModel(loaded.model.yaml, ch=3, nc=1, verbose=False)
    rebuilt.load(loaded.model, verbose=False)
    trainer = SimpleNamespace(
        model=rebuilt,
        args=SimpleNamespace(model=str(init_pt)),
    )
    audit = verify_prefusion_trainer_initialization(trainer, manifest)
    assert audit["all_checks_passed"]
    with torch.no_grad():
        trainer.model.state_dict()[CRITICAL_TENSOR_KEYS["fsm"]].view(-1)[0].add_(1.0)
    with pytest.raises(RuntimeError, match="before epoch 1"):
        verify_prefusion_trainer_initialization(trainer, manifest)


def test_safe_flops_patch_changes_only_get_flops_and_restores() -> None:
    import ultralytics.utils.torch_utils as torch_utils

    profile_path = Path(variant_config("baseline")["profile"])
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    original = torch_utils.get_flops
    restore = install_safe_prefusion_flops(profile_path)
    try:
        model = build_model("baseline").model
        assert torch_utils.get_flops(model, imgsz=640) == pytest.approx(payload["gflops"])
        assert torch_utils.model_info.__name__ == "model_info"
    finally:
        restore()
    assert torch_utils.get_flops is original


def test_run_directory_safety_preserves_crash_and_refuses_last_pt(tmp_path: Path) -> None:
    run, backup = prepare_formal_run_directory(tmp_path, "experiment")
    assert run == tmp_path / "experiment"
    assert backup is None
    run.mkdir()
    (run / "partial.txt").write_text("partial", encoding="utf-8")
    _run, backup = prepare_formal_run_directory(tmp_path, "experiment")
    assert backup is not None and (backup / "partial.txt").is_file()
    run.mkdir()
    (run / "weights").mkdir()
    (run / "weights" / "last.pt").write_bytes(b"checkpoint")
    with pytest.raises(FileExistsError, match="official resume"):
        prepare_formal_run_directory(tmp_path, "experiment")


def test_no_forbidden_architecture_or_training_features_in_new_yamls(variant_model) -> None:
    _variant, yolo = variant_model
    class_names = {module.__class__.__name__ for module in yolo.model.modules()}
    assert "FaPNOutputConv" not in class_names
    forbidden = ("DySample", "DCNv3", "DCNv4", "P2Detect", "Transformer")
    assert not any(any(token.lower() in name.lower() for token in forbidden) for name in class_names)
    assert sum(isinstance(module, DeformConv2d) for module in yolo.model.modules()) == 2
    assert all(
        isinstance(yolo.model.model[index], FaPNFeatureSelectionKeep)
        for index in FSM_INDICES
    )
    assert all(isinstance(yolo.model.model[index], FaPNAlignmentOnly) for index in ALIGN_INDICES)


def test_profile_artifacts_are_positive_and_report_fair_official_delta() -> None:
    for variant in VARIANTS:
        payload = json.loads(Path(variant_config(variant)["profile"]).read_text(encoding="utf-8"))
        assert payload["parameters"] > 0
        assert payload["trainable_parameters"] > 0
        assert payload["gflops"] > 0
        assert payload["method"]["deepcopy_used"] is False
        assert payload["comparison"]["official_yolo11n_nc1"]["parameter_delta"] > 0
        assert payload["comparison"]["official_yolo11n_nc1"]["gflops_delta"] > 0


def test_amp_probe_entrypoint_runs_fp32_cpu_fallback() -> None:
    from tools.probe_fapn_prefusion_amp import run_probe

    report = run_probe("cpu", amp=False)
    assert report["all_checks_passed"], report
    assert report["amp_executed"] is False
