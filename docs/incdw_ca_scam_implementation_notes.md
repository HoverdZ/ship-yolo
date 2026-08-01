# InceptionDW + CA-SCAM 结构实现说明

## 控制模型

- `INCDW_PLS_CA_SCAM_VGUP_150ep` 的直接控制是 `PLS_CA_SCAM_VGUP_150ep`。Backbone 中 P2/P3 的两个 `C3k2` 之外，VGUP、PLS、PAN、CA-SCAM 和 Detect 完全相同。
- `INCDW_DPLS_CA_SCAM_150ep` 的直接控制是 `R04`。Backbone 中 P2/P3 的两个 `C3k2` 之外，DPLS、PAN、CA-SCAM 和 Detect 完全相同。

## 预训练权重策略

两个模型均先构建单类目标网络，再通过语义层映射与形状一致检查从官方 `yolo11n.pt` 继承可兼容张量。训练器接管模型前会再次逐张量核对，训练参数显式禁止第二次预训练覆盖。因此 Notebook 中打印的 `Loaded/Total` 是当前组合结构的真实审计结果，而不是 Ultralytics 的另一条宽松加载日志。

## 训练与恢复

每个实验使用唯一的本地和 Drive 目录。若已有目录包含完整且匹配的实验状态与 `last.pt`，则从最近 epoch 续训；否则只能创建空目录开始新训练。该约束防止把其他结构的权重误当成同一实验恢复。

## 结果记录

完成训练后记录 Precision、Recall、mAP50、AP75、mAP50-95、最佳 epoch、参数量、GFLOPs 和官方预训练 `Loaded/Total`。缺失的正式训练指标保持为空，不从其他实验推断。

本地单类模型审计结果如下：

| 实验 | Loaded/Total | 参数量 | GFLOPs（640） |
|---|---:|---:|---:|
| `INCDW_PLS_CA_SCAM_VGUP_150ep` | 230/565 | 1,406,886 | 18.20114 |
| `INCDW_DPLS_CA_SCAM_150ep` | 230/523 | 1,337,746 | 14.82890 |
