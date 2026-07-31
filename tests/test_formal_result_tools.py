"""Result tools must preserve aliases, placeholders, and real-file provenance."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.paper_artifacts.grouped_evaluation import (
    evaluate_groups,
    make_bins,
    recommend_merged_bins,
)
from tools.paper_artifacts.per_image_evaluation import evaluate_per_image
from tools.paper_artifacts.results.builders import build
from tools.paper_artifacts.results.common import collect_metrics
from tools.paper_artifacts.results.validate_result_consistency import validate


def _manifest(root: Path, run_id: str, value: float) -> None:
    path = root / run_id / "seed_0" / "run_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": 0,
                "status": "completed",
                "initialization_weight": (
                    "yolov8n.pt" if run_id in {"R11", "R12"} else "yolo11n.pt"
                ),
                "staged_checkpoint_used": False,
                "test_used_for_selection": False,
                "git_commit": "a" * 40,
                "training": {
                    "imgsz": 640,
                    "epochs": 150,
                    "batch": 8,
                    "workers": 2,
                    "optimizer": "auto",
                    "lr0": 0.01,
                    "weight_decay": 0.0005,
                },
                "validation_metrics": {
                    "precision": value,
                    "recall": value,
                    "map50": value,
                    "map75": value,
                    "map50_95": value,
                },
                "complexity": {
                    "parameters": 1,
                    "gflops": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_missing_result_uses_explicit_placeholder(tmp_path: Path) -> None:
    row = collect_metrics(tmp_path, run_ids=["R01"])[0]
    assert row["map50_95"] == "{{PENDING_R01_MAP5095}}"
    assert row["status"] == "pending"


def test_alias_table_reads_same_canonical_manifest(tmp_path: Path) -> None:
    for run_id, value in (
        ("R00", 0.1),
        ("R02", 0.2),
        ("R04", 0.3),
        ("R10", 0.4),
    ):
        _manifest(tmp_path, run_id, value)
    paths = build("cumulative_ablation", tmp_path, tmp_path / "table")
    text = paths["csv"].read_text(encoding="utf-8-sig")
    assert "A1" in text and "R02" in text and "0.2" in text
    assert "A3" in text and "R10" in text and "0.4" in text


def test_consistency_validator_accepts_matching_real_manifests(tmp_path: Path) -> None:
    _manifest(tmp_path, "R00", 0.1)
    _manifest(tmp_path, "R02", 0.2)
    assert validate(tmp_path)["passed"]


def test_sparse_short_side_bins_produce_merge_recommendation() -> None:
    recommendation = recommend_merged_bins(
        [2.0, 3.0, 20.0, 40.0, 41.0],
        [8, 16, 32],
        minimum_instances=2,
    )
    assert recommendation["merge_notes"]
    assert len(recommendation["recommended_bins"]) < 4


def test_grouped_evaluation_reports_counts_not_fake_ap() -> None:
    records = [
        {
            "width": 640,
            "height": 640,
            "ground_truth": [
                {"short_side_at_640": 5.0},
                {"short_side_at_640": 20.0},
            ],
            "predictions": [
                {"xyxy": [0, 0, 5, 5]},
                {"xyxy": [0, 0, 40, 40]},
            ],
            "matches": [
                {
                    "prediction_index": 0,
                    "ground_truth_index": 0,
                    "iou": 1.0,
                }
            ],
        }
    ]
    rows = evaluate_groups(
        records,
        make_bins([8, 32]),
        lambda truth: truth["short_side_at_640"],
        lambda _record, prediction: min(
            prediction["xyxy"][2] - prediction["xyxy"][0],
            prediction["xyxy"][3] - prediction["xyxy"][1],
        ),
    )
    assert sum(row["tp"] for row in rows) == 1
    assert sum(row["fp"] for row in rows) == 1
    assert sum(row["fn"] for row in rows) == 1
    assert all(row["ap"] is None for row in rows)


def test_per_image_evaluation_never_predicts_the_full_split_at_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    images = root / "val" / "images"
    images.mkdir(parents=True)
    for index in range(19):
        (images / f"{index:03d}.jpg").write_bytes(b"image")
    local_yaml = tmp_path / "data.yaml"
    local_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "val": "val/images",
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        local_yaml=local_yaml,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device="cpu",
        batch=8,
        experiment_id="R01",
        run_dir=tmp_path / "run",
        tiny_short_side=16.0,
        small_short_side=32.0,
    )

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def predict(self, *, source, **_kwargs):
            values = list(source)
            self.calls.append(values)
            return iter(
                SimpleNamespace(orig_shape=(16, 16), boxes=None)
                for _path in values
            )

    model = FakeModel()
    output = evaluate_per_image(config, model)
    assert output["images"] == 19
    assert [len(call) for call in model.calls] == [8, 8, 3]
    assert max(map(len, model.calls)) <= config.batch
