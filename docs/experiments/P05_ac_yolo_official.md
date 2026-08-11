# P05：AC-YOLO 官方实现受控复现

## 实验定位

- 方法：AC-YOLO（YOLO11n 基线）
- 论文：*AC-YOLO: A lightweight ship detection model for SAR images based on YOLO11*
- 论文链接：<https://doi.org/10.1371/journal.pone.0327362>
- 作者官方仓库：<https://github.com/He-ship-sar/ACYOLO>
- 固定参考提交：`20dad8db5047add008e6eab65b032158f4a5d3e1`
- 官方仓库基础版本：Ultralytics 8.3.59
- 本项目运行版本：Ultralytics 8.4.92

本实验是新的独立对比实验 P05，不覆盖 P01。P01 的 APFAN 历史文件和结果继续保留。

## 结构与损失

1. 保留官方 YOLO11n 主干，在最深层将 `C2PSA` 替换为作者发布的
   `C2PSA_ACmix`。ACmix 并行计算局部自注意力与卷积分支，并用两个可学习标量融合。
2. 按作者发布的 `yolo11-ACmix-CCFM.yaml` 原样组织 CCFM 颈部。官方最终 YAML
   中 CCFM 是一段统一到 256 通道的 P3/P4/P5 跨尺度融合拓扑，不是单独出现的
   `CCFM` 层名。
3. 边界框回归采用作者代码中的 MPDIoU：在 IoU 上增加预测框与目标框左上角、
   右下角距离，并按对应特征图对角线平方归一化。
4. 只做兼容性改动：位置张量跟随输入设备和 dtype；MPDIoU 在 AMP 下以 float32
   计算几何项；训练仍由 Ultralytics 8.4.92 当前内核前台执行。

## 受控训练配置

- 数据源：`/content/drive/MyDrive/ship_detection/data`
- 本地副本：`/content/ship_detection/data`
- 预训练权重：官方 `yolo11n.pt`
- 本地单类模型继承审计：Loaded/Total tensors = `222/524`
- epochs：150
- imgsz：640
- batch：8
- seed：0
- workers：2
- optimizer：auto
- box/cls/dfl：7.5/0.5/1.5
- 验证和测试：`augment=False`

训练协议与其余论文对比保持一致，而不是照搬论文在 SSDD 上使用的 batch=16、SGD
等数据集相关设置。这样 AC-YOLO 与本项目最终模型的差异主要来自公开结构和 MPDIoU。

## 本地审计

- 单类模型参数量：1,845,704
- 检测步长：8/16/32
- 第 10 层：`C2PSA_ACmix`
- CCFM 头部层序列：Conv、Upsample、lateral Conv、Concat、C3k2，以及 bottom-up
  Conv/SCDown 路径，最后由 P3/P4/P5 三尺度 Detect 输出。
- CPU 模型构建、前向、反向及 MPDIoU 有限梯度检查：通过。

## 结果记录

| Precision | Recall | mAP50 | AP75 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 待训练 | 待训练 | 待训练 | 待训练 | 待训练 |

训练产物目录：
`/content/drive/MyDrive/ship_detection/paper_comparisons/P05_AC_YOLO`
