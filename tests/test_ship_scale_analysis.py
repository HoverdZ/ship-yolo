from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from tools.check_dpls_controlled_models import SPECS, check as check_dpls_models
from tools.paper_artifacts.ship_scale_analysis import (
    box_geometry,
    dilution_metrics,
    quantile_linear,
    run_analysis,
)
from tools.paper_artifacts.ship_scale_analysis.validate_scale_analysis import REQUIRED_FILES, validate_output

ROOT = Path(__file__).resolve().parents[1]


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_dataset(root: Path) -> Path:
    dataset = root / "中文数据集"
    for split in ("train", "val", "test"):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "labels").mkdir(parents=True)

    sizes = {"train": (1280, 640), "val": (640, 1280), "test": (1000, 1000)}
    for split, size in sizes.items():
        Image.new("RGB", size, "black").save(dataset / split / "images" / f"{split}_ship.png")
        Image.new("RGB", (32, 32), "black").save(dataset / split / "images" / f"{split}_empty.png")
        (dataset / split / "labels" / f"{split}_ship.txt").write_text(
            "0 0.5 0.5 0.10 0.20\n", encoding="utf-8"
        )
        (dataset / split / "labels" / f"{split}_empty.txt").write_text("", encoding="utf-8")

    val_list = dataset / "val_images.txt"
    val_list.write_text("val/images/val_empty.png\nval/images/val_ship.png\n", encoding="utf-8")
    data_yaml = root / "配置" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "train/images",
                "val": str(val_list),
                "test": str(dataset / "test" / "images"),
                "names": {0: "ship"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data_yaml


@pytest.mark.parametrize(
    ("width", "height", "width_norm", "height_norm", "expected"),
    [
        (1280, 640, 0.10, 0.20, (64.0, 64.0, 64.0, 64.0, 4096.0, 1.0)),
        (640, 1280, 0.20, 0.10, (64.0, 64.0, 64.0, 64.0, 4096.0, 1.0)),
        (1000, 1000, 0.10, 0.20, (64.0, 128.0, 64.0, 128.0, 8192.0, 2.0)),
    ],
)
def test_letterbox_box_geometry(width: int, height: int, width_norm: float, height_norm: float, expected: tuple[float, ...]) -> None:
    result = box_geometry(width, height, width_norm, height_norm, imgsz=640)
    actual = (
        result["width_640_px"],
        result["height_640_px"],
        result["short_side_640_px"],
        result["long_side_640_px"],
        result["area_640_px2"],
        result["aspect_ratio"],
    )
    assert actual == pytest.approx(expected)
    assert result["letterbox_scale"] == pytest.approx(min(640 / width, 640 / height))


def test_quantile_linear_known_values_empty_and_singleton() -> None:
    values = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    probabilities = [0.025, 0.05, 0.10, 0.50]
    expected = np.quantile(values, probabilities, method="linear")
    assert quantile_linear(values, probabilities) == pytest.approx(expected)
    assert quantile_linear([7.5], probabilities) == pytest.approx([7.5] * len(probabilities))
    with pytest.raises(ValueError, match="empty"):
        quantile_linear([], probabilities)


@pytest.mark.parametrize(
    ("short_side", "stride", "ratio", "dilution"),
    [(8.0, 4, 2.0, 0.0), (8.0, 8, 1.0, 0.0), (8.0, 16, 0.5, 50.0), (8.0, 32, 0.25, 75.0)],
)
def test_dilution_formula(short_side: float, stride: int, ratio: float, dilution: float) -> None:
    result = dilution_metrics(short_side, stride)
    assert result["sampling_intervals_spanned"] == pytest.approx(ratio)
    assert result["dilution_rate_percent"] == pytest.approx(dilution)
    assert 0 <= result["dilution_rate_percent"] <= 100


def test_full_pipeline_supports_chinese_paths_plots_and_cross_file_validation(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path)
    output = tmp_path / "分析输出"
    result = run_analysis(
        data_yaml,
        output,
        source_label="synthetic/frozen/data.yaml",
    )
    assert result["validation"]["passed"]
    assert result["validation"]["instance_count"] == 3
    for relative in REQUIRED_FILES:
        path = output / relative
        assert path.is_file() and path.stat().st_size > 0
    assert validate_output(output)["passed"]
    assert (tmp_path / "ship_scale_analysis_bundle.zip").is_file()
    raw = (output / "raw_tables" / "ship_instance_scales.csv").read_text(encoding="utf-8")
    assert str(tmp_path) not in raw
    with pytest.raises(FileExistsError):
        run_analysis(data_yaml, output, source_label="synthetic/frozen/data.yaml")


def test_repeated_runs_are_byte_identical(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path)
    first = tmp_path / "first" / "ship_scale"
    second = tmp_path / "second" / "ship_scale"
    run_analysis(data_yaml, first, source_label="synthetic/frozen/data.yaml")
    run_analysis(data_yaml, second, source_label="synthetic/frozen/data.yaml")
    assert _hashes(first) == _hashes(second)


def test_dpls_yaml_controls_and_random_weight_structure_build() -> None:
    d0 = yaml.safe_load(SPECS["D0"]["yaml"].read_text(encoding="utf-8"))
    d1 = yaml.safe_load(SPECS["D1"]["yaml"].read_text(encoding="utf-8"))
    d2 = yaml.safe_load(SPECS["D2"]["yaml"].read_text(encoding="utf-8"))
    assert d1["backbone"] == d2["backbone"]
    assert d1["head"][0][2] == "nn.Upsample"
    assert d1["head"][3][2] == "nn.Upsample"
    assert d2["head"][0][2] == "DySample"
    assert d2["head"][3][2] == "DySample"
    assert d0["head"][-1][0] == [16, 19, 22]
    assert d1["head"][-1][0] == d2["head"][-1][0] == [14, 17, 20]
    report = check_dpls_models()
    assert report["passed"]
    assert report["models"]["D0"]["strides"] == [8.0, 16.0, 32.0]
    assert report["models"]["D1"]["strides"] == report["models"]["D2"]["strides"] == [4.0, 8.0, 16.0]


def test_summary_does_not_claim_guaranteed_detection_gain(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path)
    output = tmp_path / "summary" / "ship_scale"
    run_analysis(data_yaml, output, source_label="synthetic/frozen/data.yaml")
    summary = json.loads((output / "reports" / "ship_scale_analysis_summary.json").read_text(encoding="utf-8"))
    boundary = summary["method_boundary"].lower()
    assert "must be verified" in boundary
    assert "guarantee" not in boundary
