# INCDW_DPLS_CA_SCAM_150ep

- 实验名称：YOLO11n + InceptionDW + DPLS + CA-SCAM
- 结构变化：仅 P2/P3 的 C3k2 Bottleneck 第二个卷积使用 InceptionDW；两处 DPLS DySample 上采样；P2/P3/P4 Detect 前各1个 bounded CA-SCAM；PAN 不变
- 是否使用预训练权重：是，官方 `yolo11n.pt`
- Loaded/Total tensors：`230/523`
- 参数量：1,337,746
- GFLOPs（imgsz=640）：14.82890
- epoch：150
- imgsz：640
- batch：8
- Precision：待正式训练
- Recall：待正式训练
- mAP50：待正式训练
- AP75：待正式训练
- mAP50-95：待正式训练
- 状态：代码、独立 YAML、CPU 前向/反向、结构和官方权重继承审计已通过
