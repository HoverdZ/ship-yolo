"""Generate four independent Colab notebooks for paper comparisons."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "P01": ("YOLO11s-APFAN", "P01_YOLO11s_APFAN_150ep.ipynb"),
    "P02": ("SHIP-YOLO", "P02_SHIP_YOLO_150ep.ipynb"),
    "P03": ("PMF-YOLOv8", "P03_PMF_YOLOv8_150ep.ipynb"),
    "P04": ("E-WFF Net", "P04_E_WFF_Net_150ep.ipynb"),
}


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def build_notebook(run_id: str, method: str, commit: str) -> dict:
    """Build a concise three-stage notebook with foreground training."""

    setup = f'''from google.colab import drive

# 连接当前 Google Drive；仓库已经公开，不读取也不需要 GitHub Token。
drive.mount("/content/drive", force_remount=False)

import os
import platform
import subprocess
import sys
from pathlib import Path

RUN_ID = "{run_id}"
CODE_COMMIT = "{commit}"
REPOSITORY_URL = "https://github.com/HoverdZ/ship-yolo.git"
REPOSITORY_DIR = Path("/content/ship-yolo")

# 固定和正式 YOLO11 实验相同的 Ultralytics 版本。
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.92"],
    check=True,
)


def git_run(arguments, cwd=None):
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "Git 操作失败。仓库是公开的，此错误与 Token 无关。\n"
            + result.stderr[-3000:]
        )
    return result.stdout.strip()


if REPOSITORY_DIR.exists():
    if not (REPOSITORY_DIR / ".git").is_dir():
        raise FileExistsError(
            f"{{REPOSITORY_DIR}} 已存在但不是 Git 仓库，程序不会自动删除它。"
        )
    tracked_changes = git_run(
        ["status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_DIR,
    )
    if tracked_changes:
        raise RuntimeError("现有仓库包含受跟踪文件修改，已停止以免覆盖。")
else:
    git_run(["clone", "--filter=blob:none", REPOSITORY_URL, str(REPOSITORY_DIR)])

git_run(["fetch", "--depth=1", "origin", CODE_COMMIT], cwd=REPOSITORY_DIR)
git_run(["checkout", "--detach", CODE_COMMIT], cwd=REPOSITORY_DIR)
actual_commit = git_run(["rev-parse", "HEAD"], cwd=REPOSITORY_DIR)
assert actual_commit == CODE_COMMIT, (actual_commit, CODE_COMMIT)
os.chdir(REPOSITORY_DIR)
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

import torch
import ultralytics

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
assert torch.cuda.is_available(), "没有检测到 Colab GPU，请先选择 GPU 运行时。"
print("固定代码提交：", actual_commit)
print("Python：", platform.python_version())
print("PyTorch：", torch.__version__)
print("CUDA：", torch.version.cuda)
print("Ultralytics：", ultralytics.__version__)
print("GPU：", torch.cuda.get_device_name(0))
'''

    training = '''from tools.external_baselines.remote_ship import (
    prepare_local_dataset,
    resolve_run,
    train_foreground,
)

# 云盘数据只读；用多线程 copyfile 复制到本地，并实时显示文件数和字节进度。
# 随后检查图像/标签配对、5 列 YOLO 水平框、类别范围和跨划分泄漏，
# 自动生成只指向 /content/ship_detection/data 的本地 data_runtime.yaml。
data_yaml, dataset_audit = prepare_local_dataset(
    source_root="/content/drive/MyDrive/ship_detection/data",
    destination_root="/content/ship_detection/data",
    workers=32,
)

run = resolve_run(RUN_ID)
print("实验：", run.run_id, run.method)
print("模型 YAML：", run.model_yaml)
print("官方预训练权重：", run.pretrained)
print("训练/验证/测试划分：", dataset_audit["splits"])

# 该函数先完成结构、步长、CPU 前向/反向和 Loaded/Total 审计，
# 全部通过后直接在当前内核调用官方 YOLO.train，实时显示每个 epoch。
# 若云盘中存在未完成的合法 last.pt，则自动从最近一轮恢复；不会启动训练子进程。
trained_model, train_results = train_foreground(run, data_yaml)

# 训练结束后释放训练器和优化器显存，给最终验证/测试留出空间。
import contextlib
import gc

del trained_model, train_results
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()
print(
    "训练对象已释放：",
    f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB,",
    f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB",
)
'''

    evaluation = '''from tools.external_baselines.remote_ship import (
    evaluate_best_on_val_and_test,
    resolve_run,
)

# 只加载 best.pt：先固定验证集，再评估此前封存的测试集。
# 两个划分都显式 augment=False；测试结果不参与模型选择。
run = resolve_run(RUN_ID)
summary = evaluate_best_on_val_and_test(run, data_yaml)
print("最终 val/test 汇总：", summary)
print("云盘产物目录：", run.drive_run)
'''

    note = (
        "E-WFF 原文的椭圆旋转增强依赖方向角标注；当前数据是 5 列 YOLO 水平框，"
        "因此本实验只复现网络结构，并与其他实验统一使用正式增强参数。"
        if run_id == "P04"
        else "验证集和测试集均不增强，测试集仅在训练完成后使用。"
    )
    return {
        "cells": [
            _markdown(
                f"# {run_id}：{method} 受控复现实验\n\n"
                "- 输入尺寸：640\n"
                "- 训练轮数：150\n"
                "- batch：8\n"
                "- seed：0\n"
                "- 数据源：`/content/drive/MyDrive/ship_detection/data`\n"
                f"- 说明：{note}\n"
            ),
            _markdown(
                "## 1. 挂载云盘、安装固定环境并检出公开仓库\n\n"
                "本单元格不使用 GitHub Token；若云盘或 Git 身份/访问失败，会立即停止。\n"
            ),
            _code(setup),
            _markdown(
                "## 2. 快速复制数据、审计、继承预训练权重并直接开始训练\n\n"
                "训练是当前内核中的官方 `YOLO.train(...)` 前台调用，每个 epoch 实时输出。\n"
            ),
            _code(training),
            _markdown(
                "## 3. 使用 best.pt 完成验证集和封存测试集评估\n\n"
                "训练完成后运行；输出 P、R、mAP50、AP75、mAP50-95，并同步到云盘。\n"
            ),
            _code(evaluation),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate(commit: str) -> list[Path]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit 必须是完整的 40 位小写 Git SHA。")
    output_root = ROOT / "notebooks" / "formal"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for run_id, (method, filename) in NOTEBOOKS.items():
        path = output_root / filename
        path.write_text(
            json.dumps(build_notebook(run_id, method, commit), ensure_ascii=False, indent=1)
            + "\n",
            encoding="utf-8",
        )
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    for path in generate(args.commit):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
