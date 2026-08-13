# 广域海洋遥感中的小型与极小船舶检测

本项目研究一种面向广域海洋遥感影像的可见性—尺度—上下文协同参数高效检测方法。模型以 Ultralytics YOLO11n 为基础，通过浅层空间特征提取、面向小目标的检测尺度重构、上下文校准和可见性感知输入处理，提高复杂海洋场景中小型与极小船舶的检测能力。

最终模型由以下部分组成：

- **InceptionDW**：适配至浅层 C3k2 瓶颈，用于加强 P2/P3 阶段的多尺度空间特征提取；
- **DPLS**：构建 P2–P4 检测尺度，增强小型与极小目标的表征；
- **CA-SCAM**：利用对比度感知的空间上下文校准改善低可见性目标特征；
- **VGUP**：根据图像可见性自适应调节输入增强强度。

## 目录

```text
custom_modules/   模型模块实现
experiments/      模型 YAML 与实验参数配置
model_weights/    训练权重（Git LFS）
training_logs/    训练日志
datasets/         数据集来源与使用说明
```

最终 YOLO11n 模型配置为：

```text
experiments/model_ablation/A5_inceptiondw_dpls_ca_scam_vgup.yaml
```

## 环境

- Ultralytics 8.4.92
- 输入尺寸：640 × 640
- Batch size：8
- Epochs：150
- Seed：0

自定义模型在构建前需要先注册项目模块：

```python
from custom_modules.register import register_custom_modules
from ultralytics import YOLO

register_custom_modules()
model = YOLO("experiments/model_ablation/A5_inceptiondw_dpls_ca_scam_vgup.yaml")
```

## 数据集

主要实验使用雾化增强后的 **LEVIR-Ship** 数据集。数据在公开 LEVIR-Ship 的基础上，参考 Wang et al. (2022) 的方法，以 Perlin 噪声构造薄雾、浓雾和块状雾场景。数据来源、处理方法和本地目录结构见 [`datasets/Fog-LEVIR-Ship/README.md`](datasets/Fog-LEVIR-Ship/README.md)。

受原始遥感影像使用条款限制，本仓库不直接分发图像文件。

## 参考文献

- Chen, W. et al. LEVIR-Ship: <https://github.com/WindVChen/LEVIR-Ship>
- Wang, W., Zhang, X., Sun, W., Huang, M. (2022). A novel method of ship detection under cloud interference for optical remote sensing images. *Remote Sensing*, 14(15), 3731. <https://doi.org/10.3390/rs14153731>
