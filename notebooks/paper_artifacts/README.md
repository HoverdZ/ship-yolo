# GPU 论文材料 Notebook

这三个 Notebook 只提取已训练权重的论文证据，不训练模型。依次运行：

1. `DPLS_实验材料提取_Colab.ipynb`
2. `CA-SCAM_实验材料提取_Colab.ipynb`
3. `VGUP_实验材料提取_Colab.ipynb`

主对照权重记录的训练版本为 Ultralytics 8.4.92，因此三个 Notebook 固定安装
8.4.92。可选的 `YOLO11n_baseline.pt` 记录为 8.4.109，仅作为附加参考。

运行前将正式权重放到：

```text
MyDrive/ship_detection/paper_project/论文实验材料GPU/weights/
```

所需文件：

- `YOLO11n_baseline.pt`（可选）
- `InceptionDW_PLS_CA-SCAM_VGUP_best.pt`
- `InceptionDW_DPLS_CA-SCAM_VGUP.pt`
- `InceptionDW_DPLS_SCAM_VGUP.pt`
- `InceptionDW_DPLS_CA-SCAM_best.pt`

私有仓库访问只读取 Colab 环境变量 `GITHUB_TOKEN`。不得把 Token 写进 Cell、
Notebook 或 URL。若变量不存在，Notebook 会在身份认证处停止并明确报错。

数据集来源固定为 `MyDrive/ship_detection/data`，本地副本固定为
`/content/ship_detection/data`。输出写入
`MyDrive/ship_detection/paper_project/论文实验材料GPU/输出/`，并打包为
`论文实验材料_GPU结果.zip`。
