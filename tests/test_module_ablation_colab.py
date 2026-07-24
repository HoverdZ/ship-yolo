"""Static safety and workflow tests for the Colab module-ablation entrypoints."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import colab.train_yolo11n_module_ablation as training


def test_notebook_is_valid_and_chinese_documented() -> None:
    notebook_path = ROOT / "colab/train_yolo11n_module_ablation.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "一次只训练一个实验" in text
    assert "drive.mount(\"/content/drive\")" in text
    assert "EXPERIMENT_NAME = \"yolo11n-dd\"" in text
    assert "%pip install -e /content/ship-yolo" in text
    assert "start_training(config, data_yaml)" in text
    assert "resume_training(LAST_PT)" in text
    assert "ghp_" not in text and "github.com/" + "x-access-token" not in text


def test_training_script_uses_required_copy_implementation() -> None:
    source = inspect.getsource(training)
    assert "concurrent.futures.ThreadPoolExecutor" in source
    assert "shutil.copyfile(source, destination)" in source
    assert "tqdm(total=len(source_files)" in source
    assert "copied_bytes" in source


def test_training_is_direct_and_screen_epoch_preserves_150_target() -> None:
    source = inspect.getsource(training.start_training)
    assert "model.train(" in source
    assert "subprocess" not in source
    config = training.TrainingConfig()
    assert config.epochs == 150
    assert config.screen_epoch == 80
    callback_source = inspect.getsource(training.make_screen_callback)
    assert "trainer.epoch + 1 != config.screen_epoch" in callback_source
    assert "trainer.stop = True" in callback_source
    assert "last_pt" in callback_source and "best_pt" in callback_source
    assert ".last_resume_full.pt" in callback_source
    assert "restore_screen_resume_checkpoint" in source


def test_formal_experiment_list_excludes_controls_and_invalid_combo() -> None:
    assert set(training.EXPERIMENTS) == {
        "yolo11n-c3cross",
        "yolo11n-dd",
        "yolo11n-cgfm",
        "yolo11n-inceptiondw-dd",
        "yolo11n-inceptiondw-cgfm",
    }
    assert "yolo11n-alignconcat-control" not in training.EXPERIMENTS
    assert "yolo11n-inceptiondw-c3cross" not in training.EXPERIMENTS


def test_resume_requires_last_checkpoint_and_has_no_screen_callback(tmp_path: Path) -> None:
    source = inspect.getsource(training.resume_training)
    assert "resume=True" in source
    assert "add_callback" not in source
    missing = tmp_path / "best.pt"
    try:
        training.resume_training(missing)
    except FileNotFoundError as error:
        assert "last.pt" in str(error)
    else:
        raise AssertionError("best.pt must not be accepted for formal resume")


def test_screen_checkpoint_restore_preserves_optimizer(tmp_path: Path) -> None:
    last = tmp_path / "last.pt"
    backup = tmp_path / ".last_resume_full.pt"
    marker = tmp_path / ".screen_epoch_80_done"
    torch.save({"optimizer": None, "epoch": -1}, last)
    torch.save({"optimizer": {"state": {}}, "epoch": 79}, backup)
    marker.write_text("paused once\n", encoding="utf-8")

    trainer = SimpleNamespace(save_dir=tmp_path, last=last)
    training.restore_screen_resume_checkpoint(trainer)
    restored = torch.load(last, map_location="cpu", weights_only=False)
    assert restored["optimizer"] is not None
    assert restored["epoch"] == 79
    assert not backup.exists()
