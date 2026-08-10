# 遥感船舶检测论文方法受控复现实验

## 目的与边界

本轮为论文横向对比准备四个独立实验。它们使用同一冻结数据划分、同一
Ultralytics 版本和同一训练超参数，但保留原论文声明的基础模型规模：

| ID | 方法 | 论文基础模型 | 本仓库预训练权重 | 检测步长 |
|---|---|---|---|---|
| P01 | YOLO11s-APFAN | YOLO11s | `yolo11s.pt` | 8/16/32 |
| P02 | SHIP-YOLO | YOLOv8n | `yolov8n.pt` | 8/16/32 |
| P03 | PMF-YOLOv8 | YOLOv8n | `yolov8n.pt` | 4/8/16/32 |
| P04 | E-WFF Net | YOLOv8s | `yolov8s.pt` | 8/16/32 |

作者均未提供能够直接复现论文完整检测器的公开仓库。因此这里所称“论文级复现”
是指：以论文公开的结构图、公式和消融位置为准，并优先适配有明确许可证的官方
基础模块实现。没有许可证或只有论文公式的部分由本项目独立实现，未复制第三方代码。

## 结构实现

### P01：YOLO11s-APFAN

- 论文：Wang et al., *Ship target detection algorithm in remote sensing images
  based on improved YOLO11s*, Ocean Engineering 343 (2026) 123204,
  DOI: <https://doi.org/10.1016/j.oceaneng.2025.123204>。
- 按论文总图将四个 Backbone C3K2 替换为 C3K2PKI。
- PKI 分支包含 3/5/7/9/11 深度卷积和 Context Anchor Attention；PKI 结构参考
  PKINet 官方 Apache-2.0 实现：<https://github.com/PKINet/PKINet>。
- C2PSA 后接 CAFM；Neck 中两处 P3/P4/P5 聚合节点使用 AMSFA。
- 边框回归使用 WIoU-v3 的动态非单调聚焦。

### P02：SHIP-YOLO

- 论文：Luo et al., *SHIP-YOLO: A Lightweight SAR Ship Detection Model Based
  on YOLOv8n Algorithm*, IEEE Access 12 (2024),
  DOI: <https://doi.org/10.1109/ACCESS.2024.3373893>。
- Backbone 保持 YOLOv8n，在最深 C2f 后加入 Shuffle Attention，再进入 SPPF。
- Neck 的 C2f 替换为 C2f_RepGhost；两个下采样普通卷积替换为 GhostConv；
  第一段 Top-down C2f_RepGhost 后增加第二个 Shuffle Attention。
- RepGhost 训练态分支参考作者 MIT 官方实现：
  <https://github.com/ChengpengChen/RepGhost>。
- 边框回归使用 WIoU-v3。

### P03：PMF-YOLOv8

- 论文：Chen et al., *PMF-YOLOv8: Enhanced Ship Detection Model in Remote
  Sensing Images*, Information Technology and Control 53(4) (2024),
  DOI: <https://doi.org/10.5755/j01.itc.53.4.37003>。
- Backbone 四个 C2f 改为 R-C2f，严格只把 Bottleneck 的第二个卷积换为
  receptive-field attention convolution。
- PAN 增加 P2 分支并形成四尺度输出；每个检测尺度前使用一次 FASFF，对四个
  PAN 特征计算逐像素 Softmax 权重。
- RFAConv 和 FASFF 按论文公式独立实现。RFAConv 作者仓库未声明许可证，ASFF
  参考仓库使用 GPL-3.0，因此本项目没有复制两者源码。
- 边框回归使用 Inner-MPDIoU。PMF 论文没有报告辅助框比例；本复现显式固定
  `inner_iou_ratio=0.7`，不能把该值描述成原论文公开参数。

### P04：E-WFF Net

- 论文：*E-WFF Net: An Efficient Remote Sensing Ship Detection Network Based
  on Weighted Feature Fusion*, Remote Sensing 17(6) (2025) 985,
  DOI: <https://doi.org/10.3390/rs17060985>。
- YOLOv8s 深层 Backbone 使用 DAT，随后使用 SimSPPF。
- Neck 按论文图 7 构建 P2/P4/P6/P10 残差路径，并使用 ReLU 非负权重和快速
  归一化加权融合；最终仍输出 stride 8/16/32 三个检测尺度。
- DAT 参考作者 Apache-2.0 官方实现：<https://github.com/LeapLabTHU/DAT>。
- 原文椭圆旋转增强依赖带方向角的五参数标注。当前数据标签是标准五列 YOLO
  水平框（类别与四个归一化坐标），无法忠实恢复方向角。因此 P04 只复现网络
  结构，并和其他三组统一使用正式增强策略；不估计、不伪造方向角。

## 受控训练协议

- 数据源只读：`/content/drive/MyDrive/ship_detection/data`。
- 使用 32 线程 `shutil.copyfile` 复制到 `/content/ship_detection/data`，实时显示
  文件数和已处理字节；不把训练直接指向 Drive。
- 在本地副本检查 train/val/test 目录、图像标签配对、五列 YOLO 标签、归一化
  范围、类别 0 和跨划分图像哈希泄漏；多余标签只记录，不修改云盘源数据。
- 本地自动生成 `/content/ship_detection/data_runtime.yaml`。
- Ultralytics 8.4.92，`imgsz=640`，`epochs=150`，`batch=8`，`seed=0`，
  `workers=2`，`cache=disk`。
- 其余超参数与正式实验一致：optimizer auto、lr0 0.01、lrf 0.01、momentum
  0.937、weight decay 0.0005、warmup 3、patience 100、mosaic 1、最后 10 轮
  关闭 mosaic、scale 0.5、translate 0.1、box/cls/dfl 为 7.5/0.5/1.5。
- 训练直接调用当前 Colab 内核中的官方 `YOLO.train(...)`；没有训练子进程，
  epoch 日志实时显示。
- 每轮把可恢复产物同步到
  `/content/drive/MyDrive/ship_detection/paper_comparisons/<run_name>`。
- 训练完成后固定 `best.pt`，以 `augment=False` 分别评估 val 和此前封存的 test；
  测试集不参与模型选择。

## 运行时必须保存

每组实验会保存 `args.yaml`、`results.csv`、`best.pt`、`last.pt`、训练曲线、
`pretrained_transfer_audit.json`、`val_test_metrics.csv/json` 以及 val/test 的独立
评估图。权重审计以实际单类模型为分母，输出 `Loaded/Total tensors`，不能把
YOLO 配置在 80 类时的继承数误当作实际训练继承数。

## 必要验证

`tests/test_remote_ship_reproductions.py` 覆盖：四个 YAML 构建、检测步长、关键模块
存在性、CPU 前向/反向、WIoU-v3 与 Inner-MPDIoU 梯度，以及 Notebook 无 Token、
无训练开关、前台训练和测试集固定不增强等约束。
