"""Build the official YOLO11n baseline notebook for the C1 comparison."""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "colab"
    / "conv_screening_v1"
    / "C0_YOLO11n_Official_Baseline_640.ipynb"
)
PINNED_COMMIT = "c00ba9ef3df80e07da4bcc2b20c0496e2288fa02"
BRANCH = "experiment/conv-screening-v1"


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str, tags: list[str] | None = None):
    cell = nbformat.v4.new_code_cell(text.strip() + "\n")
    if tags:
        cell.metadata["tags"] = tags
    return cell


def build():
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook.cells = [
        markdown(
            """# C0：官方 YOLO11n 同配方基线

本 Notebook 是 C1（PConv-P2/P3）的严格结构对照。模型只使用
Ultralytics 8.4.92 的官方 YOLO11n 模块；数据、epoch、imgsz、batch、
优化器策略、增强、缓存、随机种子和保存策略与 C1 完全相同。

训练 Cell 在当前 Notebook 内直接调用官方 `YOLO.train(...)`，不会把训练
放入子进程，因此每个 epoch 的官方日志会实时显示。验证集用于选择
`best.pt`，测试集默认保持封存。"""
        ),
        markdown("## 1. 唯一配置 Cell"),
        code(
            f"""
EXPERIMENT_ID = "C0_yolo11n_official"
CODE_COMMIT = "{PINNED_COMMIT}"
CODE_BRANCH = "{BRANCH}"

DRIVE_DATA_ROOT = "/content/drive/MyDrive/ship_detection/data"
LOCAL_DATA_ROOT = "/content/ship_detection/data"
DRIVE_RUNS_ROOT = "/content/drive/MyDrive/ship_detection/runs"

RUN_TRAINING = True
RUN_TEST_EVALUATION = False

# 与 C1 完全一致的受控训练设置。
EPOCHS = 150
IMGSZ = 640
BATCH = 8
WORKERS = 2
SEED = 0
CACHE = "disk"
DETERMINISTIC = False
SAVE_PERIOD = 10
COPY_WORKERS = 16
"""
        ),
        markdown("## 2. 挂载 Drive 并安装固定版本"),
        code(
            """
from google.colab import drive

drive.mount("/content/drive")
%pip install -q ultralytics==8.4.92 tqdm pyyaml

import platform
import torch
import ultralytics

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE")
print("Ultralytics:", ultralytics.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("Colab GPU 未启用，请先选择 GPU 运行时。")
""",
            ["setup"],
        ),
        markdown("## 3. 安全克隆私有仓库并检出固定 commit"),
        code(
            """
import base64
import os
import subprocess
from pathlib import Path

from google.colab import userdata

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
REPO_DIR = Path("/content/ship-yolo")
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise RuntimeError(
        "需要身份认证：Colab Secret 中没有可用的 GITHUB_TOKEN。"
        "请添加只读 Token、允许本 Notebook 访问后，重新运行此 Cell。"
    )

# Token 只进入当前 git 子进程的请求头，不写入 URL、仓库配置或 Notebook。
git_env = os.environ.copy()
git_env["GIT_TERMINAL_PROMPT"] = "0"
git_env["GIT_CONFIG_COUNT"] = "1"
git_env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
git_env["GIT_CONFIG_VALUE_0"] = (
    "AUTHORIZATION: basic "
    + base64.b64encode(f"x-access-token:{token}".encode()).decode()
)

if REPO_DIR.exists():
    if not (REPO_DIR / ".git").is_dir():
        raise FileExistsError(f"{REPO_DIR} 已存在，但不是目标 Git 仓库。")
    remote = subprocess.run(
        ["git", "-C", str(REPO_DIR), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
        env=git_env,
    ).stdout.strip()
    if remote != REPO_URL:
        raise RuntimeError(f"现有仓库 origin 不正确：{remote}")
    dirty = subprocess.run(
        ["git", "-C", str(REPO_DIR), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        env=git_env,
    ).stdout
    if dirty.strip():
        raise RuntimeError(
            "现有 /content/ship-yolo 有未提交改动。为避免误删，请重启干净运行时。"
        )
else:
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, str(REPO_DIR)],
        check=True,
        env=git_env,
    )

subprocess.run(
    ["git", "-C", str(REPO_DIR), "fetch", "--no-tags", "origin", CODE_BRANCH],
    check=True,
    env=git_env,
)
subprocess.run(
    ["git", "-C", str(REPO_DIR), "checkout", "--detach", CODE_COMMIT],
    check=True,
    env=git_env,
)
actual_commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
    env=git_env,
).stdout.strip()
assert actual_commit == CODE_COMMIT, (actual_commit, CODE_COMMIT)

del token
del git_env["GIT_CONFIG_VALUE_0"]
os.chdir(REPO_DIR)
print("Pinned repository commit:", actual_commit)
""",
            ["setup"],
        ),
        markdown(
            """## 4. 快速复制数据、解析运行状态并审计模型

复制使用 16 线程 `shutil.copyfile`，同时实时显示文件数与已处理字节。
不会拿当前云端数据与历史固定数量作比较。若同名目录已有完整、残留或
不可安全续训的产物，会保留旧目录并自动选择 `_retryN`，不会覆盖历史结果。"""
        ),
        code(
            """
import json
from pathlib import Path

from tools.conv_screening_utils import (
    ConvScreeningConfig,
    copy_dataset_to_local,
    install_trainer_handoff_guard,
    prepare_model,
    register_modules,
    resolve_run_state,
    save_preflight_reports,
)

config = ConvScreeningConfig(
    experiment_id=EXPERIMENT_ID,
    drive_data_root=DRIVE_DATA_ROOT,
    local_data_root=LOCAL_DATA_ROOT,
    drive_runs_root=DRIVE_RUNS_ROOT,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=WORKERS,
    seed=SEED,
    cache=CACHE,
    deterministic=DETERMINISTIC,
    save_period=SAVE_PERIOD,
    copy_workers=COPY_WORKERS,
)
copy_report = copy_dataset_to_local(config)
run_state = resolve_run_state(config)
print("Run state:", json.dumps(run_state, ensure_ascii=False, indent=2))

if run_state["mode"] == "new":
    prepared = prepare_model(
        config,
        official_weights="yolo11n.pt",
        run_cpu_check=True,
    )
    train_model = prepared["model"]
    preflight_dir = save_preflight_reports(
        config,
        run_state,
        prepared,
        copy_report,
    )
    install_trainer_handoff_guard(
        train_model,
        prepared["transfer"],
        preflight_dir / "trainer_handoff_report.json",
        expected_run_dir=run_state["run_dir"],
    )
    assert prepared["structure"]["expected_module"] == "C3k2"
    assert prepared["structure"]["custom_indices"] == []
    print("Loaded/Total tensors:", prepared["transfer"]["loaded_total"])
    print(
        "Loaded parameter elements:",
        f'{prepared["transfer"]["loaded_parameter_elements"]}/'
        f'{prepared["transfer"]["target_parameter_elements"]}',
    )
    print("Official-only structure audit:", prepared["structure"]["passed"])
    print("CPU forward/backward passed:", prepared["cpu_smoke"]["passed"])
else:
    register_modules()
    from ultralytics import YOLO

    train_model = YOLO(run_state["resume_checkpoint"])
    prepared = None
    transfer_path = (
        Path(run_state["run_dir"])
        / "preflight"
        / "pretrained_transfer_report.json"
    )
    if transfer_path.is_file():
        prior_transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
        print("Original Loaded/Total tensors:", prior_transfer["loaded_total"])
    print(
        "检测到有效中断 checkpoint；官方 resume 将从 epoch",
        run_state["checkpoint"]["epoch"] + 1,
        "继续。",
    )
""",
            ["preflight"],
        ),
        markdown("## 5. 正式训练：当前进程直接调用官方 API"),
        code(
            """
train_results = None
if RUN_TRAINING:
    if run_state["mode"] == "resume":
        train_results = train_model.train(resume=True)
    else:
        train_results = train_model.train(
            data=str(config.local_yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            workers=WORKERS,
            seed=SEED,
            cache=CACHE,
            deterministic=DETERMINISTIC,
            device=0,
            plots=True,
            save=True,
            save_period=SAVE_PERIOD,
            project=DRIVE_RUNS_ROOT,
            name=run_state["run_name"],
            # run_state 已保证该精确目录只含本次 preflight；若旧目录
            # 被占用则已改用 _retryN，因此这里不会覆盖历史实验。
            exist_ok=True,
            # 将上一个 Cell 中已继承并审计的内存模型交给 Trainer。
            # handoff guard 会在 epoch 1 前逐张量验证，避免权重被替换。
            pretrained=True,
        )
else:
    print("RUN_TRAINING=False：已跳过训练。")
""",
            ["formal-training"],
        ),
        markdown("## 6. 汇总验证集最佳轮次（测试集保持封存）"),
        code(
            """
from pathlib import Path

from tools.conv_screening_utils import best_metrics

if RUN_TRAINING:
    summary = best_metrics(run_state["run_dir"])
    print("Run directory:", run_state["run_dir"])
    print("Best validation epoch:", summary["best_epoch"])
    print("Precision:", summary["precision"])
    print("Recall:", summary["recall"])
    print("mAP50:", summary["map50"])
    print("mAP50-95:", summary["map50_95"])
    print("Best checkpoint:", Path(run_state["run_dir"]) / "weights" / "best.pt")
else:
    print("没有训练结果可汇总。")

if RUN_TEST_EVALUATION:
    raise RuntimeError(
        "测试集默认封存。本轮受控卷积筛选不自动运行 test；"
        "待最终结构确定后，再用独立审计流程一次性评估。"
    )
""",
            ["post-training"],
        ),
    ]
    return notebook


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build(), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
