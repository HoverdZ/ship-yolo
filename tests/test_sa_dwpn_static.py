from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.sa_dwpn_utils import all_model_yaml_paths, detect_from, load_protocol, read_yaml, spatial_positions
from tools.train_sa_dwpn_variant import ensure_new_run_dir_allowed, is_pretrain_staging_dir


EXPECTED_SPATIAL = {
    "b": [],
    "t3_only": [2],
    "o3_only": [3],
    "c_lite": [2, 3],
}


def test_all_yaml_parse_and_positions():
    for variant, path in all_model_yaml_paths().items():
        data = read_yaml(path)
        assert len([layer for layer in data["backbone"] + data["head"] if layer[2] == "SDWF"]) == 5
        assert len([layer for layer in data["backbone"] + data["head"] if layer[2] == "DWDown"]) == 1
        assert detect_from(data) == [22, 24, 26]
        assert spatial_positions(data) == EXPECTED_SPATIAL[variant]


def test_ablation_yaml_only_changes_target_spatial_flags():
    b = read_yaml(ROOT / "experiments" / "yolo11n_sa_dwpn_b.yaml")
    assert b["backbone"]
    for variant in ["t3_only", "o3_only", "c_lite"]:
        data = read_yaml(all_model_yaml_paths()[variant])
        assert data["backbone"] == b["backbone"]
        for idx, (b_layer, layer) in enumerate(zip(b["head"], data["head"]), start=len(b["backbone"])):
            if b_layer[2] == "SDWF":
                expected_spatial = idx in {21, 22} if variant == "c_lite" else idx == (21 if variant == "t3_only" else 22)
                assert layer[:3] == b_layer[:3]
                assert layer[3][:3] == b_layer[3][:3]
                assert layer[3][3] is expected_spatial
            else:
                assert layer == b_layer


def test_protocol_loads_required_values():
    protocol = load_protocol()
    training = protocol["training"]
    assert training["imgsz"] == 640
    assert training["epochs"] == 150
    assert training["batch"] == 8
    assert protocol["initialization"]["official_weights"] == "yolo11n.pt"


def test_register_module_is_importable_without_torch():
    import custom_modules.register as register

    assert hasattr(register, "register_sa_dwpn_modules")


@pytest.mark.parametrize(
    "script",
    [
        "tools/train_sa_dwpn_variant.py",
        "tools/inspect_sa_dwpn_gates.py",
        "tools/visualize_sa_dwpn_heatmaps.py",
        "tools/export_run_artifacts.py",
        "tools/test_sa_dwpn_c_lite_build.py",
        "tools/test_sa_dwpn_c_lite_forward.py",
        "tools/test_sa_dwpn_c_lite_weight_transfer.py",
    ],
)
def test_cli_help(script):
    result = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_missing_data_yaml_stops_before_training():
    result = subprocess.run(
        [
            sys.executable,
            "tools/train_sa_dwpn_variant.py",
            "--variant",
            "b",
            "--data",
            "missing-data.yaml",
            "--weights",
            "yolo11n.pt",
            "--project",
            "runs/test",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Data YAML not found" in (result.stderr + result.stdout)


def test_export_artifact_whitelist(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "args.yaml").write_text("a: 1\n", encoding="utf-8")
    (run / "results.csv").write_text("epoch\n", encoding="utf-8")
    (run / "weights").mkdir()
    (run / "weights" / "best.pt").write_text("do not copy", encoding="utf-8")
    destination = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "tools/export_run_artifacts.py", "--run-dir", str(run), "--destination", str(destination)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (destination / "args.yaml").exists()
    assert (destination / "results.csv").exists()
    assert not (destination / "weights" / "best.pt").exists()


def test_pretrain_staging_run_dir_is_allowed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "protocol.yaml").write_text("training: {}\n", encoding="utf-8")
    (run_dir / "resolved_args.json").write_text("{}\n", encoding="utf-8")
    assert is_pretrain_staging_dir(run_dir)
    ensure_new_run_dir_allowed(run_dir, exist_ok=False)


def test_existing_run_dir_with_training_artifacts_is_blocked(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "protocol.yaml").write_text("training: {}\n", encoding="utf-8")
    (run_dir / "results.csv").write_text("epoch\n", encoding="utf-8")
    assert not is_pretrain_staging_dir(run_dir)
    with pytest.raises(FileExistsError):
        ensure_new_run_dir_allowed(run_dir, exist_ok=False)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("ultralytics") is None,
    reason="torch/ultralytics not installed in this environment",
)
def test_model_build_and_dummy_forward_when_deps_available():
    subprocess.run([sys.executable, "tools/test_sa_dwpn_c_lite_build.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tools/test_sa_dwpn_c_lite_forward.py"], cwd=ROOT, check=True)
