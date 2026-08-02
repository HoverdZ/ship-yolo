"""Regression tests for the paper-material extraction workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.paper_artifacts.generate_material_notebooks import generate
from tools.paper_artifacts.gpu_material_pipeline import (
    ShortSideBin,
    grouped_counts,
    size_conditioned_ap,
)


def _record(
    *,
    truths: list[tuple[list[float], float]],
    predictions: list[tuple[list[float], float, float]],
) -> dict:
    return {
        "image": "synthetic.jpg",
        "ground_truth": [
            {"class": 0, "xyxy": box, "short_side_at_640": short_side}
            for box, short_side in truths
        ],
        "predictions": [
            {
                "class": 0,
                "xyxy": box,
                "confidence": confidence,
                "short_side_at_640": short_side,
            }
            for box, confidence, short_side in predictions
        ],
    }


def test_short_side_ap_uses_ignore_semantics_and_global_ranking() -> None:
    records = [
        _record(
            truths=[([0, 0, 6, 6], 6), ([20, 20, 40, 40], 20)],
            predictions=[
                # Higher-confidence out-of-bin GT match: ignored, not an FP.
                ([20, 20, 40, 40], 0.99, 20),
                # Higher-confidence unmatched out-of-bin prediction: ignored.
                ([50, 50, 90, 90], 0.95, 40),
                # The only target-bin GT is correctly detected.
                ([0, 0, 6, 6], 0.90, 6),
            ],
        )
    ]

    assert size_conditioned_ap(records, ShortSideBin(0, 8), 0.50) == pytest.approx(1.0)


def test_short_side_ap_penalizes_in_bin_false_positive() -> None:
    records = [
        _record(
            truths=[([0, 0, 6, 6], 6)],
            predictions=[
                ([20, 20, 26, 26], 0.95, 6),
                ([0, 0, 6, 6], 0.90, 6),
            ],
        )
    ]

    value = size_conditioned_ap(records, ShortSideBin(0, 8), 0.50)
    assert value is not None
    assert 0.49 < value < 0.51


def test_short_side_ap_empty_and_missed_groups() -> None:
    missed = [_record(truths=[([0, 0, 6, 6], 6)], predictions=[])]
    assert size_conditioned_ap(missed, ShortSideBin(0, 8), 0.50) == 0.0
    assert size_conditioned_ap(missed, ShortSideBin(32, None), 0.50) is None


def test_grouped_counts_assigns_false_positives_by_prediction_size() -> None:
    rows = grouped_counts(
        [
            _record(
                truths=[([0, 0, 6, 6], 6)],
                predictions=[
                    ([0, 0, 6, 6], 0.90, 6),
                    ([20, 20, 40, 40], 0.80, 20),
                ],
            )
        ],
        confidence_threshold=0.25,
    )
    by_group = {row["短边分组"]: row for row in rows}
    assert by_group["<8 px"]["TP"] == 1
    assert by_group["16–32 px"]["FP"] == 1
    assert by_group["<8 px"]["Recall"] == pytest.approx(1.0)


def test_generated_notebooks_are_inference_only_and_reproducible(tmp_path: Path) -> None:
    paths = generate(tmp_path)
    assert {path.name for path in paths} == {
        "DPLS_实验材料提取_Colab.ipynb",
        "CA-SCAM_实验材料提取_Colab.ipynb",
        "VGUP_实验材料提取_Colab.ipynb",
    }
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compile(code, str(path), "exec")
        assert ".train(" not in code
        assert "yolo train" not in code.lower()
        assert "RUN_TRAINING" not in code
        assert "ultralytics==8.4.109" in code
        assert "GITHUB_TOKEN" in code
        assert "shutil.copyfile" in code
        assert "ThreadPoolExecutor" in code
        assert "tqdm" in code
