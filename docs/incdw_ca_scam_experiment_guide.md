# InceptionDW + CA-SCAM 累计实验指南

## 实验范围

本组包含两个独立的150轮正式实验：

| 实验 ID | 结构变量 | Notebook |
|---|---|---|
| `INCDW_PLS_CA_SCAM_VGUP_150ep` | 完整 VGUP + P2/P3 InceptionDW + PLS + 3×bounded CA-SCAM | `notebooks/formal/INCDW_PLS_CA_SCAM_VGUP_150ep.ipynb` |
| `INCDW_DPLS_CA_SCAM_150ep` | P2/P3 InceptionDW + 2×DySample DPLS + 3×bounded CA-SCAM | `notebooks/formal/INCDW_DPLS_CA_SCAM_150ep.ipynb` |

两个实验都从官方 `yolo11n.pt` 独立初始化，不继承其他实验的 `best.pt`，并使用独立 YAML、运行目录和 Notebook。

## InceptionDW 的限定范围

- 仅替换 Backbone 中 P2、P3 后的两个 `C3k2`。
- 每个 Bottleneck 的第一个官方 `Conv` 保留。
- 仅将第二个空间卷积替换为 `InceptionDWConvBNAct`。
- P4 深层 Backbone 使用官方 `C3k2`，Neck 中所有 `C3k2` 也保持官方实现。

因此，第一个实验相对 `PLS_CA_SCAM_VGUP_150ep` 只增加 P2/P3 InceptionDW；第二个实验相对正式 `R04` 只增加同样的 P2/P3 InceptionDW。

## 共同控制变量

- 检测尺度均为 P2/P3/P4，步长 `[4, 8, 16]`。
- 三个完整 bounded CA-SCAM 位于 Detect 的三个输入之前。
- PAN 自底向上路径保持对应控制模型不变。
- `ultralytics==8.4.92`。
- `epochs=150`、`imgsz=640`、`batch=8`、`workers=2`、`optimizer=auto`。
- `lr0=0.01`、`lrf=0.01`、`momentum=0.937`、`weight_decay=0.0005`、`warmup_epochs=3`、`patience=100`。
- `mosaic=1.0`、`close_mosaic=10`、`scale=0.5`、`translate=0.1`。
- `box=7.5`、`cls=0.5`、`dfl=1.5`、`seed=0`、`cache=disk`。
- 验证集不增强，测试集保持封存。

## Colab 工作流

1. 从 Colab Secret 读取准确的 `GITHUB_TOKEN`；认证失败立即停止，固定检出提交，不执行 `git pull`。
2. 用32个线程只读复制 Drive 数据到 `/content/ship_detection/data`，同时显示文件数和字节进度，随后生成本地运行时 `data.yaml`。
3. 自动审计数据、InceptionDW 范围、PLS/DPLS、CA-SCAM、VGUP、PAN、Detect、CPU 前向/反向、复杂度和官方权重继承。
4. 检查通过后，在当前 Colab 内核中直接调用官方 `YOLO.train()`；训练不进入子进程，完整 epoch 输出实时显示。
5. 匹配的 `last.pt` 和实验状态存在时安全续训；不匹配的旧目录会拒绝复用。
6. 训练后先释放训练器和旧模型显存，再验证 `best.pt`、生成论文表格、校验和、ZIP 并原子同步到 Drive。

## 本地验证命令

```powershell
pytest -q
python tools/check_incdw_ca_scam_models.py
python tools/validate_incdw_ca_scam_notebooks.py
git diff --check
git status --short --branch
```

本地仅执行模型构建、CPU 前向/反向、权重继承和静态 Notebook 审计，不运行150轮 GPU 训练。
