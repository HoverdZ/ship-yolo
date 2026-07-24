"""YOLO11n 模块消融的 Google Colab 训练辅助代码。

本文件只提供准备、检查、训练和恢复函数；导入时不会启动正式训练。
"""

from __future__ import annotations

import base64
import concurrent.futures
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
REPO_REF = "experiment/yolo11n-crossconv-dd-cgfm"
REPO_ROOT = Path("/content/ship-yolo")
DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/ship_detection/data")
LOCAL_DATA_ROOT = Path("/content/ship_detection/data")
LOCAL_DATA_YAML = Path("/content/ship_detection/data_local.yaml")
DRIVE_RUNS_ROOT = Path("/content/drive/MyDrive/ship_detection/runs")
DRIVE_CODE_ROOT = Path("/content/drive/MyDrive/ship_detection/code")
COPY_WORKERS = min(16, max(4, (os.cpu_count() or 4) * 2))

EXPERIMENTS = {
    "yolo11n-c3cross": "experiments/yolo11n-c3cross.yaml",
    "yolo11n-dd": "experiments/yolo11n-dd.yaml",
    "yolo11n-cgfm": "experiments/yolo11n-cgfm.yaml",
    "yolo11n-inceptiondw-dd": "experiments/yolo11n-inceptiondw-dd.yaml",
    "yolo11n-inceptiondw-cgfm": "experiments/yolo11n-inceptiondw-cgfm.yaml",
}


@dataclass(frozen=True)
class TrainingConfig:
    """首批正式实验的统一训练协议。"""

    experiment_name: str = "yolo11n-dd"
    data_yaml_relative: str = "data.yaml"
    baseline_results_csv: str = ""
    weights: str = "yolo11n.pt"
    imgsz: int = 640
    epochs: int = 150
    batch: int = 8
    workers: int = 2
    seed: int = 0
    deterministic: bool = True
    amp: bool = True
    patience: int = 150
    screen_epoch: int = 80
    pause_at_screen_epoch: bool = True

    def __post_init__(self) -> None:
        if self.experiment_name not in EXPERIMENTS:
            raise ValueError(
                f"未知实验 {self.experiment_name!r}；可选值：{sorted(EXPERIMENTS)}"
            )
        if not 0 < self.screen_epoch < self.epochs:
            raise ValueError("screen_epoch 必须在 1 和 epochs-1 之间。")


def _run_git_with_temporary_header(arguments: list[str], github_token: str) -> None:
    """通过子进程环境中的临时 Header 认证，不修改 remote URL。"""
    if not github_token:
        raise ValueError("GitHub Token 不能为空。")
    credential = base64.b64encode(
        f"x-access-token:{github_token}".encode("utf-8")
    ).decode("ascii")
    child_env = os.environ.copy()
    child_env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraHeader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {credential}",
        }
    )
    try:
        subprocess.run(["git", *arguments], check=True, env=child_env)
    finally:
        child_env.pop("GIT_CONFIG_VALUE_0", None)
        credential = ""


def prepare_private_repository(github_token: str) -> str:
    """克隆或快进更新私有仓库，并返回当前 commit。"""
    if (REPO_ROOT / ".git").is_dir():
        _run_git_with_temporary_header(
            ["-C", str(REPO_ROOT), "fetch", "origin", REPO_REF],
            github_token,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "switch", REPO_REF],
            check=True,
        )
        _run_git_with_temporary_header(
            ["-C", str(REPO_ROOT), "pull", "--ff-only", "origin", REPO_REF],
            github_token,
        )
    else:
        _run_git_with_temporary_header(
            [
                "clone",
                "--branch",
                REPO_REF,
                "--single-branch",
                REPO_URL,
                str(REPO_ROOT),
            ],
            github_token,
        )
    remote = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if remote != REPO_URL:
        raise RuntimeError(f"remote URL 异常：{remote}")
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _copy_one(source: Path, destination: Path) -> tuple[int, bool]:
    """复制一个文件；worker 内部必须调用 shutil.copyfile。"""
    size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        return size, False
    shutil.copyfile(source, destination)
    return size, True


def copy_dataset_to_local(
    source_root: Path = DRIVE_DATA_ROOT,
    destination_root: Path = LOCAL_DATA_ROOT,
    workers: int = COPY_WORKERS,
) -> dict[str, Any]:
    """递归、多线程复制数据集并显示文件数和字节进度。"""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Google Drive 数据目录不存在：{source_root}")
    source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not source_files:
        raise FileNotFoundError(f"数据目录中没有文件：{source_root}")
    total_bytes = sum(path.stat().st_size for path in source_files)
    copied_files = 0
    copied_bytes = 0
    with (
        tqdm(total=len(source_files), unit="file", desc="处理文件") as file_bar,
        tqdm(total=total_bytes, unit="B", unit_scale=True, desc="处理字节") as byte_bar,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        future_to_source = {
            executor.submit(
                _copy_one,
                source,
                destination_root / source.relative_to(source_root),
            ): source
            for source in source_files
        }
        for future in concurrent.futures.as_completed(future_to_source):
            size, copied = future.result()
            copied_files += int(copied)
            copied_bytes += size if copied else 0
            file_bar.update(1)
            byte_bar.update(size)
            file_bar.set_postfix(
                copied_files=copied_files,
                copied_bytes=f"{copied_bytes:,}",
                refresh=False,
            )

    local_files = [path for path in destination_root.rglob("*") if path.is_file()]
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    yaml_files = sorted(
        path for path in local_files if path.suffix.lower() in {".yaml", ".yml"}
    )
    inventory = {
        "source_files": len(source_files),
        "local_files": len(local_files),
        "images": sum(path.suffix.lower() in image_suffixes for path in local_files),
        "labels": sum(path.suffix.lower() == ".txt" for path in local_files),
        "yaml_files": [str(path) for path in yaml_files],
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
    }
    print("本地数据清单：", inventory)
    if not yaml_files:
        raise FileNotFoundError("本地副本中没有找到数据 YAML。")
    return inventory


def create_local_data_yaml(configured_relative_path: str) -> Path:
    """复制配置指定的数据 YAML，只把 path 改成本地数据根目录。"""
    candidates = sorted(
        path
        for path in LOCAL_DATA_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    print("检测到的数据 YAML：")
    for candidate in candidates:
        print(" -", candidate)
    source_yaml = LOCAL_DATA_ROOT / configured_relative_path
    if not source_yaml.is_file():
        raise FileNotFoundError(
            "顶部 DATA_YAML_RELATIVE 指定的 YAML 不存在；"
            f"请从上方候选列表中明确选择。当前值：{configured_relative_path}"
        )
    original = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    required = {"train", "val", "test", "nc", "names"}
    missing = required.difference(original)
    if missing:
        raise KeyError(f"数据 YAML 缺少字段：{sorted(missing)}")
    local = dict(original)
    local["path"] = str(LOCAL_DATA_ROOT)
    LOCAL_DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_YAML.write_text(
        yaml.safe_dump(local, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"已生成本地数据 YAML：{LOCAL_DATA_YAML}")
    return LOCAL_DATA_YAML


def print_environment(commit: str) -> None:
    """打印论文复现所需的运行环境。"""
    import ultralytics

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未检测到 GPU"
    print("Python:", sys.version)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", gpu_name)
    print("Ultralytics:", ultralytics.__version__)
    print("仓库 commit:", commit)


METRICS = {
    "mAP50-95": ("metrics/mAP50-95(B)", "max"),
    "mAP50": ("metrics/mAP50(B)", "max"),
    "Precision": ("metrics/precision(B)", "max"),
    "Recall": ("metrics/recall(B)", "max"),
    "box loss": ("train/box_loss", "min"),
    "cls loss": ("train/cls_loss", "min"),
    "dfl loss": ("train/dfl_loss", "min"),
}


def compare_curves(
    current_csv: Path,
    baseline_csv: Path | None,
    output_dir: Path,
    screen_epoch: int,
) -> dict[str, Any]:
    """生成前 80 轮曲线、摘要表和最近十轮 mAP50-95 趋势。"""
    current = pd.read_csv(current_csv).iloc[:screen_epoch].copy()
    baseline = None
    if baseline_csv and baseline_csv.is_file():
        baseline = pd.read_csv(baseline_csv).iloc[:screen_epoch].copy()
    elif baseline_csv:
        print(f"未找到 baseline results.csv：{baseline_csv}；不会伪造比较。")

    figure, axes = plt.subplots(4, 2, figsize=(14, 18))
    rows: list[dict[str, Any]] = []
    for axis, (label, (column, direction)) in zip(axes.flat, METRICS.items()):
        if column not in current:
            axis.set_visible(False)
            continue
        current_values = pd.to_numeric(current[column], errors="coerce")
        axis.plot(current_values.index + 1, current_values, label="当前模型")
        baseline_values = None
        if baseline is not None and column in baseline:
            baseline_values = pd.to_numeric(baseline[column], errors="coerce")
            axis.plot(baseline_values.index + 1, baseline_values, label="baseline")
        best_function = np.nanmax if direction == "max" else np.nanmin
        rows.append(
            {
                "指标": label,
                "当前模型前80轮最佳值": float(best_function(current_values)),
                "baseline前80轮最佳值": (
                    float(best_function(baseline_values))
                    if baseline_values is not None
                    else np.nan
                ),
                "当前模型最近10轮均值": float(current_values.tail(10).mean()),
                "baseline最近10轮均值": (
                    float(baseline_values.tail(10).mean())
                    if baseline_values is not None
                    else np.nan
                ),
            }
        )
        axis.set_title(label)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    axes.flat[-1].set_visible(False)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "screen_epoch_curves.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "screen_epoch_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    map_column = METRICS["mAP50-95"][0]
    recent = pd.to_numeric(current[map_column], errors="coerce").dropna().tail(10)
    trend = (
        float(np.polyfit(np.arange(len(recent)), recent.to_numpy(), 1)[0])
        if len(recent) >= 2
        else float("nan")
    )
    print(f"当前模型最近10轮 mAP50-95 线性趋势：{trend:+.8f}/epoch")
    print("AP75 不在 Ultralytics 标准 results.csv 中，需结合 val 详细输出人工判断。")
    return {
        "figure": str(figure_path),
        "summary_csv": str(summary_path),
        "map50_95_recent10_slope": trend,
        "baseline_available": baseline is not None,
    }


def make_screen_callback(config: TrainingConfig):
    """创建一次性第 80 轮人工检查点 callback。"""

    def pause_after_screen_epoch(trainer) -> None:
        if trainer.epoch + 1 != config.screen_epoch:
            return
        marker = Path(trainer.save_dir) / f".screen_epoch_{config.screen_epoch}_done"
        if marker.exists():
            return
        last_pt = Path(trainer.last)
        best_pt = Path(trainer.best)
        results_csv = Path(trainer.csv)
        missing = [path for path in (last_pt, best_pt, results_csv) if not path.is_file()]
        if missing:
            raise RuntimeError(f"第 {config.screen_epoch} 轮产物缺失：{missing}")
        # Ultralytics 在训练循环退出后会为最终评估剥离 last.pt 的优化器。
        # 先保存完整 checkpoint，并在 on_train_end 恢复，保证 resume=True 真正续训。
        resume_backup = Path(trainer.save_dir) / ".last_resume_full.pt"
        shutil.copyfile(last_pt, resume_backup)
        marker.write_text("paused once\n", encoding="utf-8")
        baseline = (
            Path(config.baseline_results_csv)
            if config.baseline_results_csv.strip()
            else None
        )
        compare_curves(
            results_csv,
            baseline,
            Path(trainer.save_dir),
            config.screen_epoch,
        )
        trainer.stop = True
        print(f"\n已完成第 {config.screen_epoch} 轮并暂停。")
        print(f"请人工检查曲线；继续训练必须使用 last.pt：{last_pt}")

    return pause_after_screen_epoch


def restore_screen_resume_checkpoint(trainer) -> None:
    """最终 best.pt 评估后恢复包含优化器的完整 last.pt。"""
    backup = Path(trainer.save_dir) / ".last_resume_full.pt"
    marker_candidates = list(Path(trainer.save_dir).glob(".screen_epoch_*_done"))
    if not marker_candidates or not backup.is_file():
        return
    shutil.copyfile(backup, Path(trainer.last))
    backup.unlink()
    checkpoint = torch.load(Path(trainer.last), map_location="cpu", weights_only=False)
    if checkpoint.get("optimizer") is None:
        raise RuntimeError("恢复后的 last.pt 不含优化器，不能安全 resume。")
    print(f"已恢复可续训的完整 last.pt：{trainer.last}")


def print_training_plan(config: TrainingConfig, data_yaml: Path) -> tuple[Path, Path]:
    """构建模型并打印训练前审计，不启动训练。"""
    from custom_modules.register import register_module_ablation_modules
    from tools.module_ablation_utils import model_statistics
    from ultralytics import YOLO

    register_module_ablation_modules()
    model_yaml = REPO_ROOT / EXPERIMENTS[config.experiment_name]
    weights = REPO_ROOT / config.weights
    model = YOLO(str(model_yaml), verbose=False)
    stats = model_statistics(model, imgsz=config.imgsz)
    print("实验名：", config.experiment_name)
    print("YAML：", model_yaml)
    print("数据 YAML：", data_yaml)
    print("初始化权重：", weights)
    print("imgsz：", config.imgsz)
    print("batch：", config.batch)
    print("epochs：", config.epochs)
    print("seed：", config.seed)
    print("输出目录：", DRIVE_RUNS_ROOT / config.experiment_name)
    print("参数量：", f"{stats['parameters']:,}")
    print("GFLOPs：", f"{stats['gflops']:.3f}")
    print(f"第 {config.screen_epoch} 轮暂停：", config.pause_at_screen_epoch)
    return model_yaml, weights


def start_training(config: TrainingConfig, data_yaml: Path):
    """在当前 Notebook Python 进程直接启动一次正式实验。"""
    from custom_modules.register import register_module_ablation_modules
    from ultralytics import YOLO

    register_module_ablation_modules()
    model_yaml, weights = print_training_plan(config, data_yaml)
    model = YOLO(str(model_yaml), verbose=False)
    model.load(str(weights))
    if config.pause_at_screen_epoch:
        model.add_callback("on_fit_epoch_end", make_screen_callback(config))
        model.add_callback("on_train_end", restore_screen_resume_checkpoint)
    return model.train(
        data=str(data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        workers=config.workers,
        seed=config.seed,
        deterministic=config.deterministic,
        amp=config.amp,
        patience=config.patience,
        cache=False,
        val=True,
        save=True,
        plots=True,
        project=str(DRIVE_RUNS_ROOT),
        name=config.experiment_name,
        exist_ok=False,
        verbose=True,
    )


def resume_training(last_pt: str | Path):
    """从 last.pt 恢复优化器、调度器和 epoch；不再注册第 80 轮 callback。"""
    from custom_modules.register import register_module_ablation_modules
    from ultralytics import YOLO

    register_module_ablation_modules()
    checkpoint = Path(last_pt)
    if checkpoint.name != "last.pt" or not checkpoint.is_file():
        raise FileNotFoundError(f"必须提供存在的 last.pt：{checkpoint}")
    model = YOLO(str(checkpoint))
    return model.train(resume=True)


def sync_training_script_to_drive() -> Path:
    """把当前训练脚本副本保存到指定 Google Drive 代码目录。"""
    source = Path(__file__).resolve()
    DRIVE_CODE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = DRIVE_CODE_ROOT / source.name
    shutil.copyfile(source, destination)
    print(f"训练脚本已同步：{destination}")
    return destination


if __name__ == "__main__":
    print("请从 Colab Notebook 导入本文件；本文件不会自动启动训练。")
