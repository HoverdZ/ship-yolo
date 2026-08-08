"""D-FINE-N external-baseline preparation must remain deterministic."""

from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
import yaml
from PIL import Image

from tools.external_baselines.dfine_n import (
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_COMMIT,
    convert_yolo_to_dfine_coco,
    parse_training_log,
    parse_validator_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def _make_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (100, 80), color=(10, 20, 30)).save(
            images / f"{split}_object.jpg"
        )
        (labels / f"{split}_object.txt").write_text(
            "0 0.5 0.5 0.2 0.25\n",
            encoding="utf-8",
        )
        Image.new("RGB", (64, 64), color=(0, 0, 0)).save(
            images / f"{split}_background.jpg"
        )
        (labels / f"{split}_background.txt").write_text("", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": ["ship"],
                "nc": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_yolo_hbb_conversion_preserves_splits_and_zero_based_class(
    tmp_path: Path,
) -> None:
    data_yaml = _make_dataset(tmp_path / "dataset")
    output = tmp_path / "dfine_coco"
    report = convert_yolo_to_dfine_coco(data_yaml, output)

    assert report["category_id_policy"] == "zero_based_raw_ids_for_remap_false"
    for split in ("train", "val", "test"):
        assert report["splits"][split]["images"] == 2
        assert report["splits"][split]["instances"] == 1
        assert report["splits"][split]["background_images"] == 1
        document = json.loads(
            (output / "annotations" / f"instances_{split}.json").read_text(
                encoding="utf-8"
            )
        )
        assert document["categories"] == [
            {"id": 0, "name": "ship", "supercategory": "ship"}
        ]
        assert document["annotations"][0]["category_id"] == 0
        assert document["annotations"][0]["bbox"] == [40.0, 30.0, 20.0, 20.0]
        assert len(list((output / "images" / split).iterdir())) == 2


def test_dfine_runtime_config_locks_requested_controls() -> None:
    path = (
        ROOT
        / "experiments"
        / "external_baselines"
        / "dfine_hgnetv2_n_ship_640.yml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["epochs"] == 150
    assert config["eval_spatial_size"] == [640, 640]
    assert config["train_dataloader"]["total_batch_size"] == 8
    assert config["val_dataloader"]["total_batch_size"] == 8
    assert config["train_dataloader"]["collate_fn"]["base_size"] == 640
    assert config["train_dataloader"]["collate_fn"]["base_size_repeat"] is None
    assert config["num_classes"] == 1
    assert config["remap_mscoco_category"] is False
    assert config["HGNetv2"]["name"] == "B0"
    assert config["DFINETransformer"]["num_layers"] == 3
    assert config["optimizer"]["lr"] == 0.00005
    assert config["optimizer"]["params"][0]["lr"] == 0.000025


def test_protocol_pins_official_repository_and_foreground_training() -> None:
    protocol = yaml.safe_load(
        (
            ROOT
            / "experiments"
            / "external_baselines"
            / "dfine_n_protocol.yaml"
        ).read_text(encoding="utf-8")
    )
    assert protocol["implementation"]["commit"] == OFFICIAL_COMMIT
    assert (
        protocol["implementation"]["checkpoint_sha256"]
        == OFFICIAL_CHECKPOINT_SHA256
    )
    assert protocol["training"] == {
        "input_size": 640,
        "epochs": 150,
        "total_batch_size": 8,
        "seed": 0,
        "amp": True,
        "optimizer": "AdamW",
        "official_recipe_preserved": True,
        "batch_scaled_learning_rate": True,
        "validation_metric": "COCO AP50-95",
        "selection_checkpoint": "best validation AP50-95",
    }
    assert protocol["outputs"]["foreground_training_only"] is True
    assert protocol["outputs"]["training_subprocess_prohibited"] is True


def test_official_log_and_validator_metrics_are_parsed(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    rows = [
        {
            "epoch": 0,
            "test_coco_eval_bbox": [0.1, 0.2, 0.05, 0.01, 0.1, -1.0],
            "n_parameters": 4_000_000,
        },
        {
            "epoch": 141,
            "test_coco_eval_bbox": [0.3, 0.5, 0.25, 0.1, 0.3, -1.0],
            "n_parameters": 4_000_000,
        },
    ]
    log.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = parse_training_log(log, stop_epoch=140)
    assert result["best_epoch"] == 142
    assert result["checkpoint_name"] == "best_stg2.pth"
    assert result["map50_95"] == 0.3
    metrics = parse_validator_metrics(
        "Metrics: {'f1': 0.4, 'precision': 0.5, 'recall': 0.333, "
        "'iou': 0.6, 'TPs': 2, 'FPs': 2, 'FNs': 4}\n"
    )
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.333


def test_formal_notebook_is_pinned_and_trains_in_foreground() -> None:
    path = ROOT / "notebooks" / "formal" / "DFINE_N_Complexity_Tradeoff.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "GITHUB_TOKEN = userdata.get(\"GITHUB_TOKEN\")" in code
    assert OFFICIAL_COMMIT in code
    assert "OFFICIAL_CHECKPOINT_SHA256" in code
    assert re.search(r'SHIP_COMMIT = "[0-9a-f]{40}"', code)
    assert "dfine_train.main(train_args)" in code
    assert "RUN_TRAINING" not in code
    assert "torchrun" not in code.replace(
        "# 本单元直接运行官方训练；严禁改成 subprocess、Popen 或 torchrun。",
        "",
    )
    assert "subprocess.Popen" not in code
    assert "instances_test.json" not in (
        ROOT
        / "experiments"
        / "external_baselines"
        / "dfine_hgnetv2_n_ship_640.yml"
    ).read_text(encoding="utf-8")
    assert '"epochs": 150' in code
    assert "seed=0" in code
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        assert index > 0 and notebook.cells[index - 1].cell_type == "markdown"
        assert cell.execution_count is None
        assert cell.outputs == []
        compile(cell.source, f"{path.name}:cell-{index}", "exec")
