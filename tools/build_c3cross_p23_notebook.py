"""Generate the Colab notebook for the C3Cross P2/P3-only experiment."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab/YOLO11n_C3Cross_P23_Screening.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        markdown(
            """
# YOLO11n C3Cross-P23：AP75 审计与 30 轮筛选

## Goal

本 Notebook 只替换 P2/P3 下采样之后的 C3k2：

- layer 1、3 的普通 `3×3, stride=2 Conv` 完整保留，先做下采样与信息提取；
- layer 2、4 改为 `C3k2CrossConv`；
- P4/P5、Neck、Detect 保持官方 YOLO11n；
- 使用基线 `best.pt` 初始化普通层，再用完整 C3Cross `best.pt` 覆盖 P2/P3；
- epoch 15 不达标自动停止，最多训练 30 轮；
- 验证集只用于评估，测试集保持封存。

训练 Cell 不会自动续训，也不会覆盖已有 run。
"""
        ),
        markdown(
            """
## Setup

### 1. 安装固定版本

必须使用 Ultralytics 8.4.92。版本检查失败时立即停止。
"""
        ),
        code(
            """
%pip install -q ultralytics==8.4.92 tqdm pyyaml pandas

import ultralytics

print("Ultralytics:", ultralytics.__version__)
assert ultralytics.__version__ == "8.4.92"
"""
        ),
        markdown(
            """
### 2. 挂载 Drive，并安全获取私有仓库

在 Colab 的 Secrets 中创建 `GITHUB_TOKEN`。Token 只通过临时 HTTP Header
传给 Git 子进程，不写入 URL、Notebook、仓库或 remote。
"""
        ),
        code(
            """
from google.colab import drive, userdata

drive.mount("/content/drive")
"""
        ),
        code(
            """
import base64
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/HoverdZ/ship-yolo.git"
REPO_REF = "experiment/yolo11n-c3cross-p23"
REPO_ROOT = Path("/content/ship-yolo")

github_token = userdata.get("GITHUB_TOKEN")
if not github_token:
    raise RuntimeError("请先在 Colab Secrets 中配置 GITHUB_TOKEN")

credential = base64.b64encode(
    f"x-access-token:{github_token}".encode("utf-8")
).decode("ascii")
git_env = os.environ.copy()
git_env.update(
    {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraHeader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {credential}",
    }
)

try:
    if (REPO_ROOT / ".git").is_dir():
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "fetch", "origin", REPO_REF],
            check=True,
            env=git_env,
        )
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "switch", REPO_REF],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "pull",
                "--ff-only",
                "origin",
                REPO_REF,
            ],
            check=True,
            env=git_env,
        )
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                REPO_REF,
                "--single-branch",
                REPO_URL,
                str(REPO_ROOT),
            ],
            check=True,
            env=git_env,
        )
finally:
    git_env.pop("GIT_CONFIG_VALUE_0", None)
    credential = ""
    github_token = ""

remote = subprocess.run(
    ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
    check=True,
    text=True,
    capture_output=True,
).stdout.strip()
assert remote == REPO_URL, remote

commit = subprocess.run(
    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
    check=True,
    text=True,
    capture_output=True,
).stdout.strip()
print("Repository commit:", commit)
"""
        ),
        markdown(
            """
### 3. 配置路径

两个输入权重都来自已经完成的正式实验。P2/P3 筛选使用新目录，不修改历史结果。
"""
        ),
        code(
            """
import sys

os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.train_yolo11n_c3cross_p23 import (
    DRIVE_AUDIT_ROOT,
    DRIVE_RUNS_ROOT,
    LOCAL_DATA_YAML,
    ap75_audit,
    copy_dataset_to_local,
    create_local_data_yaml,
    hybrid_initialize,
    print_environment,
    run_p23_screening,
    run_winner_finetune,
    structure_check,
    summarize_screening_run,
    validate_checkpoint_metrics,
)

BASELINE_BEST = (
    DRIVE_RUNS_ROOT
    / "yolo11n_baseline_recheck_640"
    / "weights"
    / "best.pt"
)
C3CROSS_BEST = (
    DRIVE_RUNS_ROOT
    / "yolo11n-c3cross"
    / "weights"
    / "best.pt"
)
P23_RUN_DIR = DRIVE_RUNS_ROOT / "yolo11n-c3cross-p23"

for checkpoint in (BASELINE_BEST, C3CROSS_BEST):
    assert checkpoint.is_file(), checkpoint

print_environment(commit)
print("Baseline best:", BASELINE_BEST)
print("C3Cross best:", C3CROSS_BEST)
print("New run:", P23_RUN_DIR)
"""
        ),
        markdown(
            """
### 4. 高速复制数据集到 `/content`

固定使用 16 个 `shutil.copyfile` worker，同时显示文件数与字节进度。
同路径、同大小文件会跳过。不会比较云端与其他本地数据集的固定数量。
"""
        ),
        code(
            """
dataset_inventory = copy_dataset_to_local(workers=16)
data_yaml = create_local_data_yaml("data.yaml")
print("Training YAML:", data_yaml)
"""
        ),
        markdown(
            """
## Steps

### 5. 零训练 AP75 审计

基线和完整 C3Cross 使用相同验证集与参数。AP75 直接读取
`metrics.box.map75`，不从 mAP50-95 推算。
"""
        ),
        code(
            """
ap75_table = ap75_audit(
    data_yaml=data_yaml,
    baseline_weights=BASELINE_BEST,
    c3cross_weights=C3CROSS_BEST,
    output_csv=DRIVE_AUDIT_ROOT / "ap75_comparison.csv",
    imgsz=640,
    batch=8,
    workers=2,
    device=0,
)
ap75_table
"""
        ),
        markdown(
            """
### 6. 结构、CPU 前后向和混合权重预检

该 Cell 会明确检查 layer 1、3 仍是普通 `3×3/stride=2 Conv`，
并审计每一个目标张量来自基线还是 C3Cross P2/P3。
"""
        ),
        code(
            """
import json

preflight = structure_check(
    DRIVE_AUDIT_ROOT / "yolo11n-c3cross-p23-structure.json",
    imgsz=64,
)
print(json.dumps(preflight, ensure_ascii=False, indent=2))
assert preflight["passed"]

hybrid_model, hybrid_report = hybrid_initialize(
    BASELINE_BEST,
    C3CROSS_BEST,
)
print("Loaded/Total:", hybrid_report["loaded_total_text"])
print(
    "Baseline final tensors:",
    hybrid_report["loaded_tensors"]["baseline_final"],
)
print(
    "C3Cross P2/P3 tensors:",
    hybrid_report["loaded_tensors"]["c3cross_p23_overlay"],
)
assert hybrid_report["passed"]

del hybrid_model
"""
        ),
        markdown(
            """
### 7. 开始 30 轮筛选

若 epoch 15 之前的最佳 mAP50-95 `< 0.320`，或同一最佳轮次 Recall
`< 0.700`，训练会自动停止。新 run 已存在时拒绝覆盖。
"""
        ),
        code(
            """
screen_summary = run_p23_screening(
    data_yaml=data_yaml,
    baseline_weights=BASELINE_BEST,
    c3cross_weights=C3CROSS_BEST,
    project=DRIVE_RUNS_ROOT,
    device=0,
)
screen_summary
"""
        ),
        markdown(
            """
## Checks

### 8. 输出最终筛选结论和 P2/P3 AP75

只有同时达到 mAP50-95 ≥ 0.324、Recall ≥ 0.705、mAP50 ≥ 0.770，
才允许进入一次 20 轮微调。
"""
        ),
        code(
            """
import pandas as pd

screen_summary = summarize_screening_run(P23_RUN_DIR)
P23_BEST = P23_RUN_DIR / "weights" / "best.pt"
assert P23_BEST.is_file(), P23_BEST

p23_metrics = validate_checkpoint_metrics(
    "yolo11n-c3cross-p23",
    data_yaml,
    P23_BEST,
    imgsz=640,
    batch=8,
    workers=2,
    device=0,
)
p23_metrics_frame = pd.DataFrame([p23_metrics])
p23_metrics_output = DRIVE_AUDIT_ROOT / "ap75_c3cross_p23.csv"
p23_metrics_output.parent.mkdir(parents=True, exist_ok=True)
p23_metrics_frame.to_csv(p23_metrics_output, index=False)
print(p23_metrics_frame.to_string(index=False))
print("Saved:", p23_metrics_output)
"""
        ),
        markdown(
            """
## Next Steps

### 9. 可选：仅对晋级模型进行一次 20 轮微调

默认不运行。先确认 `summary.json` 中 `promoted_to_finetune=true`，
再把 `RUN_FINETUNE` 改为 `True`。这不是 resume，而是从胜出
`best.pt` 以低学习率新开独立实验。
"""
        ),
        code(
            """
RUN_FINETUNE = False

if RUN_FINETUNE:
    screen_summary = summarize_screening_run(P23_RUN_DIR)
    if not screen_summary["promoted_to_finetune"]:
        raise RuntimeError(
            "P2/P3-only 未达到晋级阈值，按预注册规则不得微调。"
        )
    finetune_summary = run_winner_finetune(
        data_yaml=data_yaml,
        winner_best_pt=P23_BEST,
        project=DRIVE_RUNS_ROOT,
        name="yolo11n-c3cross-p23-finetune",
        device=0,
    )
    print(finetune_summary)
else:
    print("RUN_FINETUNE=False：未启动微调。")
"""
        ),
    ]
    return notebook


def main() -> None:
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    validated = nbf.read(OUTPUT, as_version=4)
    nbf.validate(validated)
    print(f"Generated and validated: {OUTPUT}")


if __name__ == "__main__":
    main()
