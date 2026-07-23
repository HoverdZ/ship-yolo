"""Pre-training tests for SPDDown and P2 Gaussian auxiliary supervision."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.p2_gaussian_aux import (
    dense_gaussian_focal_loss,
    gaussian_heatmap_targets,
)
from custom_modules.spd import SPDDown
from tools.spddown_p2aux_utils import (
    auxiliary_gradient_report,
    build_model,
    forward_report,
    inflate_stride2_conv_to_spd,
    spd_functional_equivalence,
    structure_report,
    transfer_weights,
    yaml_scope_report,
)
from tools.train_spddown_p2aux import (
    METADATA_FILE,
    prepare_new_run_directory,
    validate_data_yaml,
)


def test_spd_phase_order_preserves_every_pixel() -> None:
    module = SPDDown(1, 1)
    x = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    rearranged = module.space_to_depth(x)
    assert rearranged.shape == (1, 4, 2, 2)
    assert torch.equal(rearranged[0, 0], x[0, 0, 0::2, 0::2])
    assert torch.equal(rearranged[0, 1], x[0, 0, 1::2, 0::2])
    assert torch.equal(rearranged[0, 2], x[0, 0, 0::2, 1::2])
    assert torch.equal(rearranged[0, 3], x[0, 0, 1::2, 1::2])


def test_spd_rejects_odd_spatial_dimensions() -> None:
    with pytest.raises(ValueError, match="even spatial"):
        SPDDown(4, 8)(torch.randn(1, 4, 15, 16))


def test_kernel_inflation_is_functionally_equivalent() -> None:
    torch.manual_seed(4)
    source = torch.randn(3, 2, 3, 3)
    target = inflate_stride2_conv_to_spd(source)
    x = torch.randn(2, 2, 18, 20)
    rearranged = torch.cat(
        [x[..., row::2, col::2] for row, col in SPDDown.phase_order],
        dim=1,
    )
    expected = torch.nn.functional.conv2d(x, source, stride=2, padding=1)
    actual = torch.nn.functional.conv2d(rearranged, target, stride=1, padding=1)
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", ["spddown", "p2_gaussian_aux"])
def test_yaml_and_structure_are_single_variable(variant: str) -> None:
    assert yaml_scope_report(variant)["all_checks_passed"]
    model = build_model(variant)
    assert structure_report(variant, model)["all_checks_passed"]


def test_spd_official_weight_mapping_and_scope() -> None:
    report = spd_functional_equivalence(ROOT / "yolo11n.pt")
    assert report["all_checks_passed"], report
    model = build_model("spddown")
    transfer = transfer_weights(model, "spddown", ROOT / "yolo11n.pt", apply=True)
    assert transfer["semantic_mapping"][0]["target_nonzero_elements"] == 36864
    assert transfer["random_initialized_parameter_keys"]
    assert not any(key.startswith("model.3.") for key in transfer["random_initialized_parameter_keys"])


def test_gaussian_targets_have_unit_centers_and_soft_neighborhood() -> None:
    batch = {
        "batch_idx": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.tensor([[0.5, 0.5, 0.08, 0.06]], dtype=torch.float32),
    }
    target = gaussian_heatmap_targets(
        batch,
        (1, 1, 32, 32),
        sigma_scale=0.25,
        min_sigma=1.0,
        max_sigma=3.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert target.shape == (1, 1, 32, 32)
    assert target.max() == 1
    assert int((target > 0.1).sum()) > 1
    logits = torch.zeros_like(target, requires_grad=True)
    loss = dense_gaussian_focal_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_auxiliary_is_training_only_and_backpropagates() -> None:
    forward = forward_report("p2_gaussian_aux", imgsz=128)
    assert forward["all_checks_passed"], forward
    assert forward["eval_auxiliary_calls"] == []
    assert forward["train_auxiliary_calls"] == [[1, 1, 32, 32]]
    gradients = auxiliary_gradient_report(imgsz=128)
    assert gradients["all_checks_passed"], gradients
    assert len(gradients["loss_items"]) == 4


def test_training_entry_is_direct_process_and_cli_help_works() -> None:
    import tools.train_spddown_p2aux as training

    source = inspect.getsource(training.run_training)
    assert "subprocess" not in source
    assert "model.train(" in source
    result = subprocess.run(
        [sys.executable, "tools/train_spddown_p2aux.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--variant" in result.stdout
    assert "--resume" in result.stdout


def test_colab_uses_getpass_without_fixed_dataset_count_gate() -> None:
    generator = (ROOT / "tools" / "build_spddown_p2aux_colab.py").read_text(encoding="utf-8")
    notebook = (
        ROOT / "notebooks" / "YOLO11n_InceptionDW_SPDDown_P2GaussianAux.ipynb"
    ).read_text(encoding="utf-8")
    for source in (generator, notebook):
        assert "getpass.getpass(" in source
        assert "EXPECTED_COUNTS" not in source
        assert "ENFORCE_EXPECTED_COUNTS" not in source
        assert "Dataset count mismatch" not in source
        assert "shutil.copyfile(" in source


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
