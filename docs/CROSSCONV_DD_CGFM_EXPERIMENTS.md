# YOLO11n CrossConv、DD 与 CGFM 模块消融

## 1. 目标与范围

本组实验服务于遥感小目标船舶检测论文，优先判断单模块是否能稳定提高最终指标，而不是继续堆叠复杂结构。首批正式实验只改变一个结构变量，保持 YOLO11n 的三尺度 Detect(P3, P4, P5)、数据划分和训练协议一致。

当前适配依据是 Ultralytics `8.4.92`。模块通过 `custom_modules/register.py` 动态注册，不修改或复制 `site-packages/ultralytics` 源码。

## 2. C3k2CrossConv

Ultralytics 8.4.92 没有独立 `CrossConv` 类，但 `C3x` 已使用等价的 CrossConv 空间结构。因此本实现不复制第二套 CrossConv，而是复用当前版本的 `Bottleneck`/`C3x`：

```text
1×3 Conv, stride=(1, 1)
→ 3×1 Conv, stride=(1, 1)
→ 通道一致且 shortcut=True 时残差相加
```

`C3k2CrossConv` 继承 `C2f`，保留双分支拆分、内部重复、拼接、最终 1×1 卷积、`shortcut`、`groups` 和 `expansion` 语义：

- `c3k=False`：每个内部重复单元是带 `(1×3, 3×1)` 卷积的 `Bottleneck`；
- `c3k=True`：每个内部重复单元使用与 C3k 层级对应的 `C3x`，不会忽略该参数；
- 只替换 Backbone P2、P3、P4、P5 四个 C3k2；Neck 中的 C3k2 全部不变。

C3CrossConv 与 InceptionDW 都改变 C3k2 内部空间提取算子，论文叙事和作用位置重叠，因此它们是竞争方案，不准备组合实验。

## 3. DD（Defect Downsampling）

> **Paper-faithful reimplementation for YOLO11**。没有可验证的作者官方完整实现，不能表述为官方代码。

DD 包含两条相加分支：

```text
输入 ─┬─ 3×3 Conv(stride=2) + BN + SiLU ─┐
      └─ DPL 信息补充分支 ────────────────┤ Add → 输出
```

DPL 在必要时只对右侧或底部补 1 个像素，使奇数高宽变为偶数，再按 `(0,0)、(1,0)、(0,1)、(1,1)` 四组交错采样。提取函数为：

```text
1×1 Conv(C→C/r) → BN → 1×1 Conv(C/r→C) → SiLU
```

代码对 `e0、e1、e2、e3` 使用同一个共享 `Extract` 实例。依据是论文公式使用同一符号 `E`，且没有作者代码证明存在四套独立权重。

四组结果拼接后执行：

```text
1×1 Conv(4C1→C2) → BN → ECA → 1×1 Conv(C2→C2) → SiLU
```

普通卷积分支和 DPL 分支都输出 `B × C2 × ceil(H/2) × ceil(W/2)`，最后逐元素相加。默认 `r=4` 是显式 YAML 参数和 YOLO11 适配决策，不冒充论文官方设置。

替换范围严格是 Backbone 中进入 P2、P3、P4、P5 的四个 stride=2 Conv；初始 Stem Conv 不变。

## 4. CGFM（Channel-Guided Fusion Module）

> **Paper-faithful reimplementation for YOLO11**。没有可验证的作者官方完整实现，不能表述为官方代码。

输入顺序固定为：

1. `X1`：P5 深层特征上采样结果；
2. `X2`：Backbone P4 浅层侧向特征。

数学结构为：

```text
X3 = Conv1×1(X1), C1→C2
X4 = Concat(X3, X2), channels=2C2
W  = Sigmoid(MLP(AvgPool(X4)) + MLP(MaxPool(X4)))
X6 = X3 * W
X7 = X2 * W
Y1 = X6 + X2
Y2 = X7 + X3
Y  = Concat(Y1, Y2), channels=2C2
```

两条池化分支分别使用 `1×1 Conv → ReLU → 1×1 Conv`，输出权重形状为 `B × C2 × 1 × 1`。默认 `reduction=16` 是显式构造参数和 YOLO11 适配决策；隐藏通道至少为 1。

只替换 Head 第一次 Top-down 融合（P5 上采样与 Backbone P4）。第二次 Top-down 和两个 Bottom-up PAN 融合仍为普通 Concat。解析器根据 CGFM 的实际 `2C2` 输出构建下游 C3k2，其最终阶段输出通道与基线一致。

## 5. AlignConcat 诊断对照

`yolo11n-alignconcat-control.yaml` 只把深层输入 1×1 对齐到浅层通道后拼接，不使用双池化权重或交叉残差。它用于区分收益来自通道对齐/压缩，还是来自 CGFM 的选择与交叉增强。该组可以构建和验证，但不属于首批正式训练。

## 6. InceptionDW 复用边界

现有已验证模型 `experiments/yolo11n_inceptiondw_c3k2_p23.yaml` 只替换 Backbone P2/P3 两个 `c3k=False` C3k2 内部 Bottleneck 的第二个 3×3 空间卷积。组合 YAML 原样复用 `C3k2_InceptionDW`：

- 不重新实现 InceptionDW；
- 不扩大到 P4/P5；
- 不改变 Neck；
- 不改变其已有权重继承方法。

## 7. 首批正式实验

| 实验 | 结构变化 | 预训练权重 | Loaded/Total tensors | epochs | imgsz | batch | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11n-c3cross | Backbone 四阶段 C3k2 → C3k2CrossConv | yolo11n.pt | 479/499 | 150 | 640 | 8 | 待正式训练 | 待正式训练 | 待正式训练 | 待正式训练 |
| yolo11n-dd | 四个 Backbone 下采样 Conv → DD | yolo11n.pt | 475/567 | 150 | 640 | 8 | 待正式训练 | 待正式训练 | 待正式训练 | 待正式训练 |
| yolo11n-cgfm | 第一次 Top-down Concat → CGFM | yolo11n.pt | 498/513 | 150 | 640 | 8 | 待正式训练 | 待正式训练 | 待正式训练 | 待正式训练 |
| yolo11n-inceptiondw-dd | 已验证 InceptionDW P2/P3 + DD | yolo11n.pt | 473/579 | 150 | 640 | 8 | 待正式训练 | 待正式训练 | 待正式训练 | 待正式训练 |
| yolo11n-inceptiondw-cgfm | 已验证 InceptionDW P2/P3 + CGFM | yolo11n.pt | 496/525 | 150 | 640 | 8 | 待正式训练 | 待正式训练 | 待正式训练 | 待正式训练 |

不能填充尚未训练得到的指标。80 轮值只用于人工筛选，也不能冒充 150 轮正式结果。

## 8. 统一训练协议

仓库现有正式协议与任务默认值一致：

```text
imgsz=640
epochs=150
batch=8
workers=2
seed=0
deterministic=True
amp=True
patience=150
```

优化器、初始学习率、调度器和数据增强未在协议文件中另行覆盖，统一沿用 Ultralytics 8.4.92 的 resolved defaults，并以每次 run 的 `args.yaml` 为事实记录。五组实验不得单独修改这些设置。验证集和测试集由原始 `data.yaml` 的 `val`/`test` 字段决定，不做训练增强，也不混入训练样本。

## 9. 本地验证结果

本地环境为 CPU-only PyTorch。所有模型使用随机 `1×3×640×640` 输入，Detect 输入保持：

```text
P3: 1×64×80×80
P4: 1×128×40×40
P5: 1×256×20×20
```

| 模型 | 参数量 | GFLOPs | 权重张量匹配率 | CPU 640 前向 |
| --- | ---: | ---: | ---: | --- |
| yolo11n-c3cross | 2,497,408 | 6.381 | 479/499 (95.99%) | 通过 |
| yolo11n-dd | 2,946,680 | 7.741 | 475/567 (83.77%) | 通过 |
| yolo11n-cgfm | 2,653,296 | 6.679 | 498/513 (97.08%) | 通过 |
| yolo11n-inceptiondw-dd | 2,941,764 | 7.641 | 473/579 (81.69%) | 通过 |
| yolo11n-inceptiondw-cgfm | 2,648,380 | 6.579 | 496/525 (94.48%) | 通过 |
| yolo11n-alignconcat-control | 2,640,720 | 6.668 | 498/505 (98.61%) | 通过 |

权重继承只接受同名且同形状张量；不 reshape、不切片、不做语义错误映射。新模块保持随机初始化。完整机器可读报告在 `reports/module_ablation_validation.json`。

## 10. Google Colab 使用

入口：

- `colab/train_yolo11n_module_ablation.ipynb`
- `colab/train_yolo11n_module_ablation.py`

Drive 路径：

```text
数据源：/content/drive/MyDrive/ship_detection/data
Colab 本地副本：/content/ship_detection/data
本地 data YAML：/content/ship_detection/data_local.yaml
训练结果：/content/drive/MyDrive/ship_detection/runs
脚本副本：/content/drive/MyDrive/ship_detection/code
```

Notebook 步骤：

1. 设置单个 `EXPERIMENT_NAME`；
2. 挂载 Drive；
3. 通过 `getpass` 输入 Token，用临时 HTTP Header 获取私有仓库；
4. `%pip install -e /content/ship-yolo`；
5. 多线程复制数据集，打印文件、图片、标签和 YAML 清单；
6. 从明确配置的数据 YAML 生成本地副本，只改 `path`；
7. 打印环境、commit、模型参数量、GFLOPs 和训练计划；
8. 用户明确后手动运行正式训练单元格。

不比较云端数据集与其他本地版本的固定数量。多个 YAML 时必须在顶部明确设置 `DATA_YAML_RELATIVE`，不能猜测。

## 11. 第 80 轮暂停与恢复

初始训练始终设置 `epochs=150`。`on_fit_epoch_end` 在第 80 轮验证、`results.csv` 写入以及 `last.pt`/`best.pt` 保存后触发：

- marker 文件保证每个 run 只暂停一次；
- 暂停时暂存包含优化器的完整 `last.pt`，并在 Ultralytics 最终 best 评估后恢复它，避免框架自动剥离优化器导致无法 resume；
- 生成七类指标/损失曲线；
- 若 baseline CSV 存在，输出双方前 80 轮最佳值和最近 10 轮均值；
- 计算当前模型最近 10 轮 mAP50-95 线性趋势；
- 设置 `trainer.stop=True`，等待人工判断。

找不到 baseline 时会明确提示，只输出当前模型数据，不伪造比较。AP75 不在标准 `results.csv` 中，需要结合验证详细输出人工判断。

继续时使用独立单元格：

```python
model = YOLO("/path/to/last.pt")
results = model.train(resume=True)
```

恢复阶段不再注册暂停 callback；必须使用 `last.pt` 恢复优化器、学习率调度器和 epoch，不能从 `best.pt` 重新初始化或微调。

## 12. 复现注意事项

- 当前本地验证未启动正式训练，也未测试 GPU 速度；
- DD/CGFM 是论文忠实思路的 YOLO11 适配，不是作者官方代码；
- 数据集、缓存、`runs/`、权重、Token 和临时认证文件不得提交；
- 每次正式 run 必须保留 `args.yaml`、`results.csv`、checkpoint 和实验记录；
- 训练结束后把最终 Precision、Recall、mAP50、mAP50-95 回填到实验记录，不能用筛选值替代。
