"""Generate three focused Colab notebooks for GPU-only paper evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "paper/extract-experiment-materials"
ULTRALYTICS_VERSION = "8.4.92"


def markdown(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CELL = f"""
# 作用：挂载Google Drive、安装与正式checkpoint一致的Ultralytics版本，并取得分析代码。
from google.colab import drive
drive.mount('/content/drive')

import base64
import os
import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--quiet',
     'ultralytics=={ULTRALYTICS_VERSION}', 'tqdm', 'pyyaml', 'pandas', 'matplotlib'],
    check=True,
)

import ultralytics
assert ultralytics.__version__ == '{ULTRALYTICS_VERSION}', ultralytics.__version__
print('Ultralytics:', ultralytics.__version__)

TOKEN = os.environ.get('GITHUB_TOKEN')
if not TOKEN:
    raise RuntimeError('缺少GITHUB_TOKEN环境变量；请完成身份认证后重新运行。本程序不会要求在Cell中粘贴Token。')

REPO_DIR = Path('/content/ship-yolo')
REPO_URL = 'https://github.com/HoverdZ/ship-yolo.git'
auth = base64.b64encode(f'x-access-token:{{TOKEN}}'.encode()).decode()
git_auth = ['git', '-c', f'http.https://github.com/.extraheader=AUTHORIZATION: basic {{auth}}']

if (REPO_DIR / '.git').is_dir():
    subprocess.run(git_auth + ['-C', str(REPO_DIR), 'fetch', 'origin', '{BRANCH}'], check=True)
    subprocess.run(['git', '-C', str(REPO_DIR), 'switch', '{BRANCH}'], check=True)
    subprocess.run(git_auth + ['-C', str(REPO_DIR), 'pull', '--ff-only', 'origin', '{BRANCH}'], check=True)
else:
    if REPO_DIR.exists():
        raise RuntimeError(f'{{REPO_DIR}} 已存在但不是Git仓库，请人工确认；程序不会自动删除。')
    subprocess.run(git_auth + ['clone', '--branch', '{BRANCH}', '--single-branch', REPO_URL, str(REPO_DIR)], check=True)

sys.path.insert(0, str(REPO_DIR))
print('仓库提交:', subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', 'HEAD'], text=True).strip())
"""


DATA_CELL = """
# 作用：多线程复制Drive数据集到Colab本地；文件数与字节数均实时显示，重复运行时只补齐变化文件。
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm.auto import tqdm
import yaml

DRIVE_DATA = Path('/content/drive/MyDrive/ship_detection/data')
LOCAL_DATA = Path('/content/ship_detection/data')
if not DRIVE_DATA.is_dir():
    raise FileNotFoundError(DRIVE_DATA)
LOCAL_DATA.mkdir(parents=True, exist_ok=True)

source_files = sorted(path for path in DRIVE_DATA.rglob('*') if path.is_file())
total_bytes = sum(path.stat().st_size for path in source_files)
workers = min(32, max(8, (os.cpu_count() or 4) * 4))

def copy_one(source):
    relative = source.relative_to(DRIVE_DATA)
    destination = LOCAL_DATA / relative
    size = source.stat().st_size
    if destination.is_file() and destination.stat().st_size == size:
        return size, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return size, True

copied = 0
with tqdm(total=len(source_files), desc='数据集文件', unit='个', dynamic_ncols=True) as file_bar, \
     tqdm(total=total_bytes, desc='数据集字节', unit='B', unit_scale=True, unit_divisor=1024, dynamic_ncols=True) as byte_bar, \
     ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(copy_one, path) for path in source_files]
    for future in as_completed(futures):
        size, changed = future.result()
        copied += int(changed)
        file_bar.update(1)
        byte_bar.update(size)
        file_bar.set_postfix(本次复制=copied, 已存在=len(source_files) - copied)

def split_images(root, names):
    candidates = []
    for name in names:
        candidates.extend([root / 'images' / name, root / name / 'images'])
    return next((path for path in candidates if path.is_dir()), None)

train_images = split_images(LOCAL_DATA, ['train'])
val_images = split_images(LOCAL_DATA, ['val', 'valid', 'validation'])
test_images = split_images(LOCAL_DATA, ['test'])
if train_images is None or val_images is None:
    raise RuntimeError(f'无法识别训练/验证图片目录：{LOCAL_DATA}')

LOCAL_YAML = Path('/content/ship_detection/data.yaml')
payload = {
    'path': str(LOCAL_DATA),
    'train': str(train_images.relative_to(LOCAL_DATA)),
    'val': str(val_images.relative_to(LOCAL_DATA)),
    'nc': 1,
    'names': {0: 'ship'},
}
if test_images is not None:
    payload['test'] = str(test_images.relative_to(LOCAL_DATA))
LOCAL_YAML.parent.mkdir(parents=True, exist_ok=True)
LOCAL_YAML.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding='utf-8')

print('本地数据集:', LOCAL_DATA)
print('本地data.yaml:', LOCAL_YAML)
print(payload)
"""


COMMON_CONFIG = """
# 作用：固定推理协议和Drive输出路径；本Notebook不训练任何模型。
from pathlib import Path
import torch

assert torch.cuda.is_available(), '本任务依赖GPU；请将Colab运行时切换为GPU。'
GPU_ROOT = Path('/content/drive/MyDrive/ship_detection/paper_project/论文实验材料GPU')
WEIGHT_DIR = GPU_ROOT / 'weights'
OUTPUT_ROOT = GPU_ROOT / '输出'
GPU_ROOT.mkdir(parents=True, exist_ok=True)
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

IMGSZ = 640
CONFIDENCE_FLOOR = 0.001
COUNT_CONFIDENCE = 0.25
NMS_IOU = 0.7
DEVICE = 0
BATCH = 8
print('GPU:', torch.cuda.get_device_name(0))
print('输出根目录:', OUTPUT_ROOT)
"""


def _cache_expression(label: str, output_var: str) -> str:
    safe = label.replace("/", "_").replace(" ", "_")
    return f"""
def ensure_cache(label, weights, output_dir, split='val'):
    expected = output_dir / f'{{label.replace("/", "_").replace(" ", "_")}}_{{split}}逐图预测缓存.json'
    if expected.is_file():
        import json
        from tools.paper_artifacts.gpu_material_pipeline import sha256_file
        metadata = json.loads(expected.read_text(encoding='utf-8'))
        valid = (
            metadata.get('model') == label
            and metadata.get('split') == split
            and int(metadata.get('imgsz', -1)) == IMGSZ
            and float(metadata.get('confidence_floor', -1)) == CONFIDENCE_FLOOR
            and float(metadata.get('nms_iou', -1)) == NMS_IOU
            and metadata.get('weights_sha256') == sha256_file(weights)
        )
        if valid:
            print('复用已通过权重与协议审计的预测缓存:', expected)
            return expected
        print('缓存与当前权重或推理协议不一致，将重新生成:', expected)
    return generate_prediction_cache(
        weights=weights,
        data_yaml=LOCAL_YAML,
        output_dir=output_dir,
        model_label=label,
        split=split,
        imgsz=IMGSZ,
        confidence_floor=CONFIDENCE_FLOOR,
        nms_iou=NMS_IOU,
        device=DEVICE,
        batch=BATCH,
    )
"""


def dpls_cells() -> list[dict[str, Any]]:
    return [
        markdown(
            """
            # DPLS论文实验材料提取

            只使用已训练的 `best.pt`，完成PLS/DPLS短边分组评价、短边条件AP、真实Detect输入特征响应、逐图预测缓存和代表案例候选。不会启动训练。

            运行前请将下列权重放入 `MyDrive/ship_detection/paper_project/论文实验材料GPU/weights/`：

            - `YOLO11n_baseline.pt`（可选参考）
            - `InceptionDW_PLS_CA-SCAM_VGUP_best.pt`
            - `InceptionDW_DPLS_CA-SCAM_VGUP.pt`

            主对照PLS/DPLS权重的checkpoint训练版本为Ultralytics 8.4.92；可选基线权重记录为8.4.109，但本Notebook在统一8.4.92推理环境下读取，并在缓存中保留权重SHA256。
            """
        ),
        code(SETUP_CELL),
        code(DATA_CELL),
        code(
            COMMON_CONFIG
            + """
PLS_WEIGHT = WEIGHT_DIR / 'InceptionDW_PLS_CA-SCAM_VGUP_best.pt'
DPLS_WEIGHT = WEIGHT_DIR / 'InceptionDW_DPLS_CA-SCAM_VGUP.pt'
BASELINE_WEIGHT = WEIGHT_DIR / 'YOLO11n_baseline.pt'
for path in (PLS_WEIGHT, DPLS_WEIGHT):
    if not path.is_file():
        raise FileNotFoundError(path)
DPLS_OUTPUT = OUTPUT_ROOT / '01_DPLS'
DPLS_OUTPUT.mkdir(parents=True, exist_ok=True)
print('DPLS输出:', DPLS_OUTPUT)
"""
        ),
        code(
            """
# 作用：运行DPLS全部GPU证据提取，并在Drive生成可续用预测缓存和结果ZIP。
import pandas as pd
from tools.paper_artifacts.gpu_material_pipeline import (
    compare_prediction_caches,
    evaluate_dpls_size_groups,
    export_dpls_feature_comparison,
    generate_prediction_cache,
    package_gpu_results,
)
"""
            + _cache_expression("PLS", "DPLS_OUTPUT")
            + """
pls_cache = ensure_cache('PLS', PLS_WEIGHT, DPLS_OUTPUT, split='val')
dpls_cache = ensure_cache('DPLS', DPLS_WEIGHT, DPLS_OUTPUT, split='val')
caches = {'PLS': pls_cache, 'DPLS': dpls_cache}
if BASELINE_WEIGHT.is_file():
    caches['YOLO11n'] = ensure_cache('YOLO11n', BASELINE_WEIGHT, DPLS_OUTPUT, split='val')

if test_images is not None:
    ensure_cache('PLS', PLS_WEIGHT, DPLS_OUTPUT, split='test')
    ensure_cache('DPLS', DPLS_WEIGHT, DPLS_OUTPUT, split='test')
    if BASELINE_WEIGHT.is_file():
        ensure_cache('YOLO11n', BASELINE_WEIGHT, DPLS_OUTPUT, split='test')
    print('test逐图缓存已生成；分组评价和案例筛选仍只使用val。')

evaluate_dpls_size_groups(caches, DPLS_OUTPUT, confidence_threshold=COUNT_CONFIDENCE)
candidate_csv = compare_prediction_caches(
    left_cache=pls_cache,
    right_cache=dpls_cache,
    output_dir=DPLS_OUTPUT,
    prefix='DPLS',
    left_label='PLS',
    right_label='DPLS',
    confidence_threshold=COUNT_CONFIDENCE,
)
candidates = pd.read_csv(candidate_csv)
example_image = candidates.iloc[0]['源路径']
export_dpls_feature_comparison(
    pls_weights=PLS_WEIGHT,
    dpls_weights=DPLS_WEIGHT,
    image=example_image,
    output_dir=DPLS_OUTPUT,
    imgsz=IMGSZ,
    device=DEVICE,
)
zip_path = package_gpu_results(OUTPUT_ROOT, GPU_ROOT / '论文实验材料_GPU结果.zip')
print('DPLS材料完成:', DPLS_OUTPUT)
print('结果ZIP:', zip_path)
"""
        ),
    ]


def ca_scam_cells() -> list[dict[str, Any]]:
    return [
        markdown(
            """
            # CA-SCAM论文实验材料提取

            固定 InceptionDW + DPLS + VGUP，对SCAM与CA-SCAM进行同协议逐图比较，导出真实局部对比度、空间校准、残差和beta证据。不会训练。

            权重目录：`MyDrive/ship_detection/paper_project/论文实验材料GPU/weights/`

            - `InceptionDW_DPLS_SCAM_VGUP.pt`
            - `InceptionDW_DPLS_CA-SCAM_VGUP.pt`
            """
        ),
        code(SETUP_CELL),
        code(DATA_CELL),
        code(
            COMMON_CONFIG
            + """
SCAM_WEIGHT = WEIGHT_DIR / 'InceptionDW_DPLS_SCAM_VGUP.pt'
CA_SCAM_WEIGHT = WEIGHT_DIR / 'InceptionDW_DPLS_CA-SCAM_VGUP.pt'
for path in (SCAM_WEIGHT, CA_SCAM_WEIGHT):
    if not path.is_file():
        raise FileNotFoundError(path)
CA_OUTPUT = OUTPUT_ROOT / '02_CA-SCAM'
CA_OUTPUT.mkdir(parents=True, exist_ok=True)
print('CA-SCAM输出:', CA_OUTPUT)
"""
        ),
        code(
            """
# 作用：生成SCAM/CA-SCAM预测缓存、代表案例和不改变forward的内部机制可视化。
import pandas as pd
from tools.paper_artifacts.gpu_material_pipeline import (
    compare_prediction_caches,
    export_ca_scam_debug,
    generate_prediction_cache,
    package_gpu_results,
)
"""
            + _cache_expression("SCAM", "CA_OUTPUT")
            + """
scam_cache = ensure_cache('SCAM', SCAM_WEIGHT, CA_OUTPUT, split='val')
ca_cache = ensure_cache('CA-SCAM', CA_SCAM_WEIGHT, CA_OUTPUT, split='val')
if test_images is not None:
    ensure_cache('SCAM', SCAM_WEIGHT, CA_OUTPUT, split='test')
    ensure_cache('CA-SCAM', CA_SCAM_WEIGHT, CA_OUTPUT, split='test')
    print('test逐图缓存已生成；代表案例筛选仍只使用val。')
candidate_csv = compare_prediction_caches(
    left_cache=scam_cache,
    right_cache=ca_cache,
    output_dir=CA_OUTPUT,
    prefix='CA-SCAM',
    left_label='SCAM',
    right_label='CA-SCAM',
    confidence_threshold=COUNT_CONFIDENCE,
)
candidates = pd.read_csv(candidate_csv)
example_image = candidates.iloc[0]['源路径']
export_ca_scam_debug(
    weights=CA_SCAM_WEIGHT,
    image=example_image,
    output_dir=CA_OUTPUT,
    imgsz=IMGSZ,
    device=DEVICE,
)
zip_path = package_gpu_results(OUTPUT_ROOT, GPU_ROOT / '论文实验材料_GPU结果.zip')
print('CA-SCAM材料完成:', CA_OUTPUT)
print('结果ZIP:', zip_path)
"""
        ),
    ]


def vgup_cells() -> list[dict[str, Any]]:
    return [
        markdown(
            """
            # VGUP论文实验材料提取

            固定 InceptionDW + DPLS + CA-SCAM，对无VGUP/有VGUP模型生成预测缓存、代表案例、输入处理路径和全验证集门控统计。不会训练。

            权重目录：`MyDrive/ship_detection/paper_project/论文实验材料GPU/weights/`

            - `InceptionDW_DPLS_CA-SCAM_best.pt`
            - `InceptionDW_DPLS_CA-SCAM_VGUP.pt`
            """
        ),
        code(SETUP_CELL),
        code(DATA_CELL),
        code(
            COMMON_CONFIG
            + """
NO_VGUP_WEIGHT = WEIGHT_DIR / 'InceptionDW_DPLS_CA-SCAM_best.pt'
VGUP_WEIGHT = WEIGHT_DIR / 'InceptionDW_DPLS_CA-SCAM_VGUP.pt'
for path in (NO_VGUP_WEIGHT, VGUP_WEIGHT):
    if not path.is_file():
        raise FileNotFoundError(path)
VGUP_OUTPUT = OUTPUT_ROOT / '03_VGUP'
VGUP_OUTPUT.mkdir(parents=True, exist_ok=True)
print('VGUP输出:', VGUP_OUTPUT)
"""
        ),
        code(
            """
# 作用：运行VGUP逐图对照、真实输入处理路径和全验证集门控统计。
import pandas as pd
from tools.paper_artifacts.gpu_material_pipeline import (
    analyze_vgup_validation_gates,
    compare_prediction_caches,
    export_vgup_debug,
    generate_prediction_cache,
    package_gpu_results,
)
"""
            + _cache_expression("无VGUP", "VGUP_OUTPUT")
            + """
no_vgup_cache = ensure_cache('无VGUP', NO_VGUP_WEIGHT, VGUP_OUTPUT, split='val')
vgup_cache = ensure_cache('有VGUP', VGUP_WEIGHT, VGUP_OUTPUT, split='val')
if test_images is not None:
    ensure_cache('无VGUP', NO_VGUP_WEIGHT, VGUP_OUTPUT, split='test')
    ensure_cache('有VGUP', VGUP_WEIGHT, VGUP_OUTPUT, split='test')
    print('test逐图缓存已生成；代表案例筛选和门控统计仍只使用val。')
candidate_csv = compare_prediction_caches(
    left_cache=no_vgup_cache,
    right_cache=vgup_cache,
    output_dir=VGUP_OUTPUT,
    prefix='VGUP',
    left_label='无VGUP',
    right_label='有VGUP',
    confidence_threshold=COUNT_CONFIDENCE,
)
candidates = pd.read_csv(candidate_csv)
example_image = candidates.iloc[0]['源路径']
export_vgup_debug(
    weights=VGUP_WEIGHT,
    image=example_image,
    output_dir=VGUP_OUTPUT,
    imgsz=IMGSZ,
    device=DEVICE,
)
analyze_vgup_validation_gates(
    weights=VGUP_WEIGHT,
    data_yaml=LOCAL_YAML,
    output_dir=VGUP_OUTPUT,
    split='val',
    imgsz=IMGSZ,
    device=DEVICE,
    batch=BATCH,
)
zip_path = package_gpu_results(OUTPUT_ROOT, GPU_ROOT / '论文实验材料_GPU结果.zip')
print('VGUP材料完成:', VGUP_OUTPUT)
print('结果ZIP:', zip_path)
"""
        ),
    ]


NOTEBOOKS = {
    "DPLS_实验材料提取_Colab.ipynb": dpls_cells,
    "CA-SCAM_实验材料提取_Colab.ipynb": ca_scam_cells,
    "VGUP_实验材料提取_Colab.ipynb": vgup_cells,
}


def generate(output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, factory in NOTEBOOKS.items():
        path = output / name
        path.write_text(
            json.dumps(notebook(factory()), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "notebooks" / "paper_artifacts"),
    )
    args = parser.parse_args()
    for path in generate(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
