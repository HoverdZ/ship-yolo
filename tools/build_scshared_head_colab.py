"""生成带中文说明的 SCSharedHead 分阶段训练 Colab 笔记本。"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "YOLO11n_InceptionDW_SCSharedHead.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "accelerator": "GPU",
    "colab": {"name": OUTPUT.name, "provenance": []},
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}
notebook["cells"] = [
    markdown(
        """# YOLO11n-InceptionDW-SCSharedHead 分阶段训练

本笔记本只训练一个变量明确的实验：保留已经有效的 InceptionDW
浅层骨干和原始 YOLO11 Neck，仅把 Detect(P3, P4, P5) 替换为尺度校准
共享检测头。

训练从第一轮起就采用 150 轮学习率调度，但会在第 80 轮自动暂停并保存
`stage80_resume.pt`。分析 80 轮结果后，再决定是否从第 81 轮继续到第
150 轮。正式训练直接运行在当前笔记本内核中，保证实时输出不会消失。"""
    ),
    code(
        """# Cell 1：挂载 Google Drive，并定义本次实验的固定路径
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
SOURCE_DATA_ROOT = DRIVE_ROOT / "ship_detection" / "data"
DRIVE_RUNS = DRIVE_ROOT / "ship_detection" / "runs_scshared_head"
DRIVE_REPORTS = DRIVE_ROOT / "ship_detection" / "preflight_scshared_head"

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
BRANCH = "experiment/inceptiondw-scshared-head"
REPO = Path("/content/ship-yolo")
LOCAL_DATA_ROOT = Path("/content/datasets/ship_detection")
RUN_NAME = "yolo11n_inceptiondw_scshared_head_640"

for path in (DRIVE_RUNS, DRIVE_REPORTS):
    path.mkdir(parents=True, exist_ok=True)

print("实验名称：", RUN_NAME)
print("云端数据集：", SOURCE_DATA_ROOT)
print("训练输出：", DRIVE_RUNS / RUN_NAME)"""
    ),
    code(
        """# Cell 2：使用 getpass 隐藏输入 GitHub Token，并安全克隆私有分支
import getpass
import os
import stat
import subprocess

token = getpass.getpass(
    "请输入 GitHub Token（输入隐藏，仅供本单元格调用 git，不会写入文件）："
).strip()
if not token:
    raise ValueError("GitHub Token 不能为空。")

# 使用临时 GIT_ASKPASS 传递凭据，避免把 Token 写进仓库 URL 或项目文件。
askpass = Path("/content/.ship_yolo_askpass.py")
askpass.write_text(
    "#!/usr/bin/env python3\\n"
    "import os, sys\\n"
    "prompt = sys.argv[1] if len(sys.argv) > 1 else ''\\n"
    "print('x-access-token' if 'Username' in prompt else os.environ['SHIP_GITHUB_TOKEN'])\\n",
    encoding="utf-8",
)
askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
git_env = {
    **os.environ,
    "GIT_ASKPASS": str(askpass),
    "GIT_TERMINAL_PROMPT": "0",
    "SHIP_GITHUB_TOKEN": token,
}

def run_git(arguments, *, authenticated=False):
    completed = subprocess.run(
        ["git", *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=git_env if authenticated else None,
    )
    output = (completed.stdout + completed.stderr).replace(token, "[已隐藏]").strip()
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(map(str, arguments))} 执行失败，"
            f"退出码 {completed.returncode}：\\n{output or '没有诊断输出。'}"
        )
    if output:
        print(output)
    return completed.stdout.strip()

try:
    if REPO.exists() and not (REPO / ".git").is_dir():
        if any(REPO.iterdir()):
            raise RuntimeError(
                f"{REPO} 已存在，但不是目标 Git 仓库。"
                "请先移动该无关目录，再重新运行本单元格。"
            )
        REPO.rmdir()

    if REPO.exists():
        remote = run_git(["-C", str(REPO), "remote", "get-url", "origin"])
        accepted = {REPO_URL.rstrip("/"), REPO_URL.removesuffix(".git").rstrip("/")}
        if remote.rstrip("/") not in accepted:
            raise RuntimeError(f"拒绝操作无关仓库，其远程地址为：{remote}")
        run_git(
            ["-C", str(REPO), "fetch", "origin", BRANCH],
            authenticated=True,
        )
        run_git(
            ["-C", str(REPO), "checkout", "-B", BRANCH, f"origin/{BRANCH}"]
        )
    else:
        run_git(
            [
                "-c", "credential.helper=", "clone", "--branch", BRANCH,
                "--single-branch", REPO_URL, str(REPO),
            ],
            authenticated=True,
        )
finally:
    # 无论成功还是报错，都立即清除内存变量和临时凭据脚本。
    token = ""
    git_env.pop("SHIP_GITHUB_TOKEN", None)
    os.environ.pop("SHIP_GITHUB_TOKEN", None)
    askpass.unlink(missing_ok=True)

PINNED_COMMIT = run_git(["-C", str(REPO), "rev-parse", "HEAD"])
run_git(["-C", str(REPO), "checkout", "--detach", PINNED_COMMIT])
print("本次训练固定提交：", PINNED_COMMIT)"""
    ),
    code(
        """# Cell 3：安装并核对固定版本的训练环境
%pip install -q ultralytics==8.4.92

import sys
sys.path.insert(0, str(REPO))

import torch
import ultralytics
from custom_modules.register import register_custom_modules

assert ultralytics.__version__ == "8.4.92", ultralytics.__version__
register_custom_modules()
if not torch.cuda.is_available():
    raise RuntimeError("正式训练必须使用 Colab GPU 运行时。")

print("Python：", sys.version.split()[0])
print("Torch：", torch.__version__)
print("Ultralytics：", ultralytics.__version__)
print("GPU：", torch.cuda.get_device_name(0))"""
    ),
    code(
        """# Cell 4：使用多线程 shutil.copyfile 把云端数据复制到 Colab 本地，并实时显示进度
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import time
from tqdm.auto import tqdm
import yaml

if not SOURCE_DATA_ROOT.is_dir():
    raise FileNotFoundError(f"找不到云端数据集目录：{SOURCE_DATA_ROOT}")

source_files = sorted(
    path
    for path in SOURCE_DATA_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() != ".cache"
)
if not source_files:
    raise RuntimeError(f"数据集目录中没有文件：{SOURCE_DATA_ROOT}")

source_sizes = {path: path.stat().st_size for path in source_files}
total_bytes = sum(source_sizes.values())

# 这里只清理 Colab 临时磁盘中的旧副本，不会删除或修改 Google Drive 源数据。
if LOCAL_DATA_ROOT.exists():
    shutil.rmtree(LOCAL_DATA_ROOT)
LOCAL_DATA_ROOT.mkdir(parents=True)

def copy_one(source: Path):
    relative = source.relative_to(SOURCE_DATA_ROOT)
    destination = LOCAL_DATA_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return relative, source_sizes[source]

workers = min(32, max(4, (os.cpu_count() or 4) * 2))
started = time.perf_counter()
copied_files = 0
copied_bytes = 0

with (
    tqdm(
        total=len(source_files),
        desc="文件进度",
        unit="个",
        dynamic_ncols=True,
        mininterval=0.2,
        position=0,
    ) as file_bar,
    tqdm(
        total=total_bytes,
        desc="字节进度",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=0.2,
        position=1,
    ) as byte_bar,
    ThreadPoolExecutor(max_workers=workers) as pool,
):
    futures = [pool.submit(copy_one, path) for path in source_files]
    for future in as_completed(futures):
        _, size = future.result()
        copied_files += 1
        copied_bytes += size
        file_bar.update(1)
        byte_bar.update(size)
        elapsed = max(time.perf_counter() - started, 1e-6)
        file_bar.set_postfix(
            线程数=workers,
            速度=f"{copied_bytes / elapsed / (1024**2):.1f} MiB/s",
            refresh=False,
        )

# 只核对本次动态扫描得到的文件数和字节数，不使用固定的数据集数量门槛。
local_files = [
    path
    for path in LOCAL_DATA_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() != ".cache"
]
local_bytes = sum(path.stat().st_size for path in local_files)
if len(local_files) != copied_files or local_bytes != copied_bytes:
    raise RuntimeError(
        "复制后的动态核对失败："
        f"源端 {copied_files} 个文件/{copied_bytes} 字节，"
        f"本地 {len(local_files)} 个文件/{local_bytes} 字节。"
    )

print(
    f"已使用 {workers} 个线程复制 {copied_files} 个文件，"
    f"共 {copied_bytes / 1024**3:.2f} GiB。"
)

if (LOCAL_DATA_ROOT / "train" / "images").is_dir():
    split_paths = {split: f"{split}/images" for split in ("train", "val", "test")}
elif (LOCAL_DATA_ROOT / "images" / "train").is_dir():
    split_paths = {split: f"images/{split}" for split in ("train", "val", "test")}
else:
    raise RuntimeError("数据集必须包含 train/images 或 images/train 目录结构。")

LOCAL_DATA_YAML = LOCAL_DATA_ROOT / "data.yaml"
LOCAL_DATA_YAML.write_text(
    yaml.safe_dump(
        {
            "path": str(LOCAL_DATA_ROOT),
            "train": split_paths["train"],
            "val": split_paths["val"],
            "test": split_paths["test"],
            "nc": 1,
            "names": {0: "ship"},
        },
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)
print(LOCAL_DATA_YAML.read_text(encoding="utf-8"))"""
    ),
    code(
        """# Cell 5：在训练前执行结构、前向、计算量和权重继承审计
from tools.check_scshared_head import main as run_preflight

report = run_preflight(
    [
        "--weights", str(REPO / "yolo11n.pt"),
        "--imgsz", "640",
        "--output-dir", str(DRIVE_REPORTS / PINNED_COMMIT),
    ]
)
assert report["all_checks_passed"]

print("参数量：", report["statistics"]["parameters"])
print("GFLOPs：", report["statistics"]["gflops_at_imgsz"])
transfer = report["weight_transfer"]
print(
    "从 yolo11n.pt 继承的状态张量：",
    f"{transfer['loaded_state_tensors']}/{transfer['total_state_tensors']}",
)
print(
    "安全映射的原生 Detect 输出张量：",
    transfer["mapped_native_detect_output_tensors"],
)
print(
    "继承的目标参数元素比例：",
    f"{transfer['loaded_target_parameter_element_ratio']:.4%}",
)
print("全部训练前检查已经通过。")"""
    ),
    markdown(
        """## 第一阶段：训练到第 80 轮后自动暂停

下面的单元格从一开始就声明 150 轮调度，因此前 80 轮可以与已有
InceptionDW 模型的前 80 轮曲线比较。第 80 轮会额外保存未剥离优化器
状态的 `stage80_resume.pt`。"""
    ),
    code(
        """# Cell 6：开始 80 轮筛选训练；训练直接运行在当前内核中，以便实时显示日志
from tools.train_scshared_head import TrainingRequest, run_training

stage_request = TrainingRequest(
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    weights=str(REPO / "yolo11n.pt"),
    name=RUN_NAME,
    total_epochs=150,
    stage_epochs=80,
    imgsz=640,
    batch=8,
    workers=2,
    device="0",
    seed=0,
    optimizer="auto",
    resume=False,
)
stage_result = run_training(stage_request)

stage_checkpoint = DRIVE_RUNS / RUN_NAME / "weights" / "stage80_resume.pt"
if not stage_checkpoint.is_file():
    raise FileNotFoundError(
        f"第 80 轮结束后没有生成必需的续训检查点：{stage_checkpoint}"
    )
print("80 轮筛选训练已完成：", stage_checkpoint)"""
    ),
    code(
        """# Cell 7：仅在第 80 轮之前因断线中止时使用本单元格
from tools.train_scshared_head import TrainingRequest, run_training

last_checkpoint = DRIVE_RUNS / RUN_NAME / "weights" / "last.pt"
stage_checkpoint = DRIVE_RUNS / RUN_NAME / "weights" / "stage80_resume.pt"

if stage_checkpoint.is_file():
    raise RuntimeError(
        "第 80 轮正式检查点已经存在，不应再运行“中断恢复”单元格。"
    )
if not last_checkpoint.is_file():
    raise FileNotFoundError(f"找不到中断训练检查点：{last_checkpoint}")

interrupted_request = TrainingRequest(
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    name=RUN_NAME,
    total_epochs=150,
    stage_epochs=80,
    resume=True,
    resume_checkpoint=str(last_checkpoint),
)
interrupted_result = run_training(interrupted_request)"""
    ),
    code(
        """# Cell 8：汇总 80 轮曲线；本单元格只展示数据，不自动替你作继续训练的决定
import pandas as pd

results_csv = DRIVE_RUNS / RUN_NAME / "results.csv"
if not results_csv.is_file():
    raise FileNotFoundError(results_csv)

frame = pd.read_csv(results_csv)
metric_columns = [
    "epoch",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

if len(frame) < 80:
    print(f"当前训练尚未完成：已记录 {len(frame)}/80 轮。")
else:
    stage = frame.iloc[:80].copy()
    metric = "metrics/mAP50-95(B)"
    best_index = stage[metric].astype(float).idxmax()
    last20_mean = stage[metric].astype(float).tail(20).mean()

    print("第 1～80 轮的最佳指标：")
    display(stage.loc[[best_index], metric_columns])
    print(f"最后 20 轮 mAP50-95 均值：{last20_mean:.6f}")
    print("最后 20 轮完整曲线：")
    display(stage[metric_columns].tail(20))
    print(
        "已有 InceptionDW 前 80 轮参考："
        "best mAP50-95≈0.30168，最后 20 轮均值≈0.27472。"
    )
    print("请把本单元格结果交给 Codex 分析后，再决定是否运行续训单元格。")"""
    ),
    markdown(
        """## 第二阶段：经分析确认后，从第 81 轮续训到第 150 轮

不要在第 80 轮结束后立即运行。确认有继续价值后，将下一单元格中的
`CONFIRM_CONTINUE_TO_150` 改为 `True`。这会恢复优化器、学习率调度器、
AMP 和 EMA 状态，不是重新开始一个 70 轮微调。"""
    ),
    code(
        """# Cell 9：可选续训；只有确认通过后才把开关改为 True
CONFIRM_CONTINUE_TO_150 = False
if not CONFIRM_CONTINUE_TO_150:
    raise RuntimeError(
        "续训开关仍为 False。请先分析 80 轮结果，再决定是否继续。"
    )

from tools.train_scshared_head import TrainingRequest, run_training

stage_checkpoint = DRIVE_RUNS / RUN_NAME / "weights" / "stage80_resume.pt"
continue_request = TrainingRequest(
    data=str(LOCAL_DATA_YAML),
    project=str(DRIVE_RUNS),
    name=RUN_NAME,
    total_epochs=150,
    stage_epochs=80,
    resume=True,
    resume_checkpoint=str(stage_checkpoint),
)
final_result = run_training(continue_request)
print("已完成从第 81 轮到第 150 轮的正式续训。")"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
