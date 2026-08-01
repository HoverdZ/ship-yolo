# PLS + SCAM 系列正式实验指南

## 实验范围

本组包含四个独立的 150 epoch 正式实验：

| 实验 ID | 结构变量 | Notebook |
|---|---|---|
| `PLS_CA_SCAM_150ep` | PLS + 3×bounded CA-SCAM | `notebooks/formal/PLS_CA_SCAM_150ep.ipynb` |
| `PLS_SCAM_150ep` | PLS + 3×原始 SCAM | `notebooks/formal/PLS_SCAM_150ep.ipynb` |
| `PLS_CA_SCAM_VGUP_150ep` | 完整 VGUP + PLS + 3×bounded CA-SCAM | `notebooks/formal/PLS_CA_SCAM_VGUP_150ep.ipynb` |
| `PLS_CA_SCAM_ERUP_150ep` | ERUP + PLS + 3×bounded CA-SCAM | `notebooks/formal/PLS_CA_SCAM_ERUP_150ep.ipynb` |

四个实验均从官方 `yolo11n.pt` 独立初始化，不继承 R01、DPLS 或其他消融实验的 `best.pt`。每个实验使用独立 YAML、运行目录和 Notebook，不覆盖历史结果。

## 共同控制变量

- 两个上采样节点均为 PLS 的 `nn.Upsample(scale_factor=2, mode="nearest")`，不存在 DySample。
- 检测尺度均为 P2/P3/P4，步长 `[4, 8, 16]`。
- SCAM 或 CA-SCAM 均位于3个 Detect 输入之前。
- 两条 PAN 自底向上路径保持与 R01 PLS 一致。
- `ultralytics==8.4.92`。
- `epochs=150`、`imgsz=640`、`batch=8`、`workers=2`、`optimizer=auto`。
- `lr0=0.01`、`lrf=0.01`、`momentum=0.937`、`weight_decay=0.0005`、`warmup_epochs=3`、`patience=100`。
- `mosaic=1.0`、`close_mosaic=10`、`scale=0.5`、`translate=0.1`。
- `box=7.5`、`cls=0.5`、`dfl=1.5`、`seed=0`、`cache=disk`。

## Colab 工作流

1. 从 Colab Secret 读取准确的 `GITHUB_TOKEN`，认证失败立即停止；使用固定提交，不执行 `git pull`。
2. 用32个线程只读复制 Drive 数据集到 `/content/ship_detection/data`，实时显示文件数和字节进度，并在本地生成运行时 `data.yaml`。
3. 自动审计数据、PLS 上采样、SCAM 类型和位置、PAN、Detect、CPU 前向/反向、复杂度及官方权重继承。
4. 所有检查通过后，在当前 Colab 内核中直接调用官方 `YOLO.train()`；训练严禁进入子进程，epoch 输出实时显示。
5. 若存在匹配的 `last.pt` 与状态文件则从最近一轮续训；完整实验拒绝覆盖。
6. 训练后先释放训练器和优化器显存，再加载 `best.pt` 完成固定验证、逐图统计、复杂度、校验和、ZIP 和 Drive 原子同步。测试集保持封存。

## 本地验证命令

```powershell
pytest -q
python tools/check_pls_scam_models.py
python tools/validate_pls_scam_notebooks.py
git diff --check
git status --short --branch
```

本地环境只验证模型构建、CPU 前向/反向和权重继承，不执行150轮 GPU 训练。
