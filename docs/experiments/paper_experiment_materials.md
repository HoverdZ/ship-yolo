# Ocean Engineering 论文实验材料提取

本流程只读取已经训练完成的 `best.pt`、checkpoint 内嵌训练结果和
`results.csv`，不会训练或续训模型。Windows 端负责权重身份审计、日志整理、
复杂度复算、训练曲线和已有尺度材料归档；逐图预测、特征响应和门控统计由
Colab GPU 完成。

## Windows CPU 流程

```powershell
python tools/paper_artifacts/build_experiment_materials.py `
  --results-dir <正式权重与CSV目录> `
  --material-root <论文实验材料输出目录> `
  --scale-materials <已确认尺度分析目录> `
  --repo-root .
```

生成的材料目录只包含五个一级分类文件夹，分类文件夹内部保持平坦。CSV 使用
UTF-8 BOM。复杂度由当前真实 checkpoint/YAML 重新构建并计算；checkpoint
训练版本和本地审计版本分别记录。

材料索引源 CSV 由上一步生成。正式 XLSX 使用 `@oai/artifact-tool` 构建：

```text
tools/paper_artifacts/build_material_index.mjs
```

## 权重身份规则

身份审计同时使用：文件名、checkpoint 训练参数、内嵌模型结构、仓库 YAML
匹配、模块签名、Git 信息和 checkpoint 内嵌 `train_results` 与候选 CSV 的
逐列数值匹配。无法得到足够证据时保留“部分确认”或“待人工确认”，不使用
四舍五入指标反推身份。

## Colab GPU 流程

Notebook 位于 `notebooks/paper_artifacts/`：

1. `DPLS_实验材料提取_Colab.ipynb`
2. `CA-SCAM_实验材料提取_Colab.ipynb`
3. `VGUP_实验材料提取_Colab.ipynb`

三个 Notebook 均固定相同推理协议，调用 Ultralytics 官方 `YOLO.predict()`，
不含 `.train()`。Drive 数据集会通过多线程 `shutil.copyfile` 复制到 Colab
本地，并同步显示文件数和字节进度。预测缓存同时保存 JSON 和 CSV，缓存只有
在权重 SHA256 与推理协议完全一致时才会复用。

Drive 权重目录：

```text
/content/drive/MyDrive/ship_detection/paper_project/论文实验材料GPU/weights/
```

需要的权重文件：

- `YOLO11n_baseline.pt`（DPLS 中可选参考）
- `InceptionDW_PLS_CA-SCAM_VGUP_best.pt`
- `InceptionDW_DPLS_CA-SCAM_VGUP.pt`
- `InceptionDW_DPLS_SCAM_VGUP.pt`
- `InceptionDW_DPLS_CA-SCAM_best.pt`

推荐按上面的 Notebook 顺序运行。统一输出目录为：

```text
/content/drive/MyDrive/ship_detection/paper_project/论文实验材料GPU/输出/
```

每次运行结束都会更新 `论文实验材料_GPU结果.zip`。DPLS/PLS 的正式 Detect
输入是 P2、P3、P4，不存在作为 Detect 输入的 P5，因此可视化程序不会伪造
P5 特征图。

## 验证

人工构造的短边条件 AP 测试覆盖：目标尺度组 TP、目标尺度组高置信度 FP、
非目标尺度 GT/检测忽略语义、空尺度组和固定阈值分组计数。CA-SCAM 与 VGUP
调试接口在运行时报告正常 forward 和 debug forward 的 `max_abs_diff` 与
`mean_abs_diff`，差异超出 `1e-6` 时立即失败。
