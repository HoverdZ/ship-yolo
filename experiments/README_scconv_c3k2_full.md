# YOLO11n-SCConv-C3k2-Full

## 实验目的

本实验只将官方 YOLO11n Backbone 四个 `C3k2` 内部负责空间特征提取的
3×3、stride=1 普通卷积替换为 SCConv；外层 CSP/C2f 拓扑、全部 1×1
卷积、阶段间 stride=2 下采样卷积、SPPF、C2PSA、Neck、Detect 和损失均不变。

模型配置来自 Ultralytics 8.4.92 的 `cfg/models/11/yolo11.yaml`，使用
`n: [0.50, 0.25, 1024]` 缩放。实验分支基于 `main@52d8a0c`。

## SCConv 来源与本项目适配

论文：

- Jiafeng Li, Ying Wen, Lianghua He, “SCConv: Spatial and Channel
  Reconstruction Convolution for Feature Redundancy,” CVPR 2023.
- 论文页面：<https://openaccess.thecvf.com/content/CVPR2023/html/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.html>

公开参考：

- 非官方 PyTorch 实现：<https://github.com/cheng-haha/ScConv>

`custom_modules/scconv.py` 是面向本项目重新整理的复现，不是论文作者官方代码。
它保留论文的 SRU→CRU 顺序和默认设置：

- SRU GroupNorm groups：4
- gate threshold：0.5
- CRU split alpha：0.5
- squeeze ratio：2
- GWC groups：2
- GWC kernel：3

相对公开参考实现，本版本规范了 `gate_threshold`、`squeeze_ratio` 拼写，
加入了通道、分组、零通道和奇偶性检查；归一化使用
`sqrt(variance + eps)`，gamma 归一化分母带 `eps=1e-5` 保护；自定义
`GroupBatchnorm2d` 的 affine weight 以 1 初始化，而不是随机初始化。所有
张量操作继承输入的 device/dtype，没有写死 CUDA。

公开参考代码只实现同通道输入输出；论文的 CRU 参数公式同时定义了 C1 和
C2。本适配按该公式允许 `out_channels`，默认仍等于输入通道。这样
`C3k2(c3k=False)` 内部 Bottleneck 的 `e=0.5` 隐藏通道比例可以原样保留，
无需在 SCConv 外新增 1×1 投影。

Ultralytics `Conv` 实际包含 Conv2d、BatchNorm2d 和 SiLU。为了只改变空间
卷积，本适配用 SCConv 替换内部 Conv2d，同时保留对应的 BatchNorm2d 和
SiLU；C3k2/C3k 外层 1×1 `Conv` 的类、属性名和参数路径保持不变。

## 精确结构范围

配置文件：`experiments/yolo11n_scconv_c3k2_full.yaml`

- Backbone 层 2、4、6、8：`C3k2` → `C3k2_SCConv`
- Backbone 层 0、1、3、5、7：原始 stride=2 `Conv` 不变
- Neck 层 13、16、19、22：保持原始 `C3k2`
- Head stride=2 层 17、20：保持原始 `Conv`
- Detect：仍接收层 16、19、22，即 P3/P4/P5

在 n 缩放下，四个节点内部的 ScConv 数量依次为 2、2、4、4。

## 本地 CPU 检查

在仓库根目录安装与实验一致的 Ultralytics 后运行：

```bash
pip install "ultralytics==8.4.92" pytest pyyaml
python tools/check_scconv_units.py
python tools/inspect_scconv_model.py --imgsz 640 --device cpu
python tools/compare_scconv_baseline.py --imgsz 640
python tools/inspect_scconv_weight_transfer.py --weights yolo11n.pt
python -m pytest tests/test_scconv.py tests/test_sa_dwpn_static.py
```

`inspect_scconv_model.py` 默认执行 batch=1、640×640 的 CPU eval 前向，
检查输出有限值、Detect 三层特征尺寸和 stride。

## 官方预训练权重继承

训练不是完全随机初始化。`inspect_scconv_weight_transfer.py` 和训练入口会：

1. 构建官方 `yolo11n.pt`；
2. 构建本实验 YAML；
3. 按参数名和 shape 继承兼容权重；
4. 输出张量与参数元素继承比例、missing/unexpected/shape mismatch；
5. 分组核对下采样卷积、Backbone C3k2 外层 1×1、SPPF、C2PSA、Neck 和 Detect。

新 SCConv 内部权重无法继承原 3×3 Conv2d 权重，这是预期行为。保留下来的
外层 1×1、未修改模块，以及被替换 Conv 块原有且 shape 兼容的 BN 参数可继承。
若 Neck 或 Detect 不是完整继承，工具会将其标为异常。

## Colab 安装与训练

```bash
git clone --branch experiment/scconv-c3k2-full \
  https://github.com/HoverdZ/ship-yolo.git
cd ship-yolo
pip install "ultralytics==8.4.92" pytest pyyaml

python tools/check_scconv_units.py
python tools/inspect_scconv_model.py --imgsz 640 --device cpu
python tools/inspect_scconv_weight_transfer.py --weights yolo11n.pt

python tools/train_scconv_c3k2_full.py \
  --data /content/path/to/ship/data.yaml \
  --weights yolo11n.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 8 \
  --workers 2 \
  --device 0 \
  --project /content/path/to/runs/scconv_c3k2_full \
  --name yolo11n_scconv_c3k2_full_640 \
  --seed 0
```

训练入口会先注册自定义模块、构建实验 YAML、加载官方权重、打印并保存继承
报告和模型规模，然后才调用训练。非 resume 模式若目标目录已有训练产物会
立即停止，避免覆盖。恢复同一运行：

```bash
python tools/train_scconv_c3k2_full.py \
  --data /content/path/to/ship/data.yaml \
  --project /content/path/to/runs/scconv_c3k2_full \
  --name yolo11n_scconv_c3k2_full_640 \
  --resume
```

只验证训练前准备而不启动训练，可附加 `--dry-run`。

## 训练参数与预期文件

默认正式配置为 imgsz=640、epochs=150、batch=8、workers=2、device=0、
seed=0；数据集类别由传入的单类 ship 数据 YAML 决定。训练目录应包含：

- `resolved_args.json`
- `weight_transfer.json`
- `args.yaml`
- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- confusion matrix
- PR curve

## 必须在 Colab/GPU 补做

本地不进行完整训练或 GPU 性能测试。正式报告还需记录：

- 参数量与 GFLOPs 复核
- 峰值 GPU 显存
- 单 epoch 和总训练耗时
- 单图 GPU/CPU 推理延迟
- Precision、Recall、mAP50、mAP50-95、可用时的 AP75
- `best.pt`、`last.pt`、`results.csv`、`results.png`
- 混淆矩阵和 PR 曲线

GFLOPs 降低不等于真实延迟降低；SCConv 包含归一化、门控、分组卷积、
池化、拆分和拼接，实际速度必须在目标 GPU 上测量。
