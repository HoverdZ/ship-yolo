# YOLO11n-InceptionDW-SCSharedHead 设计与训练方案

## 1. 实验目标

本实验保留已经验证有效的 `C3k2_InceptionDW` 浅层骨干，仅替换
YOLO11 的三尺度检测头。Neck、P2 检测层、损失函数、数据增强策略与
训练超参数保持不变，以验证共享检测头是否能在降低参数量的同时改善
小目标船舶的分类和定位。

实验属于根据论文结构和公式完成的独立复现与适配，不是作者官方代码。
PMS-YOLO 的公开仓库当前未提供 SCDH 模块实现，TAD-YOLO 论文也没有
给出可直接移植到本项目的完整 DY-CTMH 代码。

## 2. 论文依据

### 2.1 PMS-YOLO 的 SCDH

SCDH 在 P3、P4、P5 之间共享卷积，使用 GroupNorm 代替 BatchNorm，
然后以独立输出层产生分类与回归结果，并为不同尺度设置可学习缩放。
其 SeaShips 消融中，SCDH 单独将 mAP50-95 从 0.806 提升到 0.821，
参数量从 3.01M 降至 2.36M。

### 2.2 TAD-YOLO 的 DY-CTMH

DY-CTMH 首先使用两层共享的 `3×3 Conv-GN` 提取多尺度共同特征，
随后加入任务交互、分类空间概率图及定位 DCNv2。其独立消融将
mAP50-95 从 0.801 提升到 0.815，参数量从 3.1M 降至 1.7M。

## 3. 本项目结构

每个 P3/P4/P5 特征依次执行：

1. 独立 `1×1 Conv-GN-SiLU` 输入适配，将通道统一为 64；
2. 共用同一组两层 `3×3 Conv-GN-SiLU`；
3. 乘以该尺度独立的正数校准系数 `exp(log_scale_i)`；
4. 使用独立 `1×1` 卷积输出 64 通道 DFL 回归分布；
5. 使用独立 `1×1` 卷积输出单类别分类 logits。

检测尺度仍为 P3/P4/P5，步长仍为 8/16/32，DFL 解码、标签分配、
损失函数和 NMS 均沿用 Ultralytics YOLO11。

## 4. 与两篇论文的区别

| 项目 | PMS-SCDH | TAD-DY-CTMH | 本项目 SCSharedDetect |
|---|---|---|---|
| 输入通道处理 | 论文图示以共享头为主 | 共享多特征提取 | 每个尺度先独立 1×1 适配 |
| 共享卷积 | 有 | 两层 3×3 Conv-GN | 两层 3×3 Conv-GN |
| 归一化 | GN | GN | GN |
| 尺度校准 | 可学习 Scale | 任务调制为主 | 每尺度独立正数 Scale |
| 分类空间图 | 无 | 有 | 第一版不使用 |
| 定位 DCNv2 | 无 | 有 | 第一版不使用 |
| Neck 改动 | MAFPN | ASDFDPN | 无 |
| 输出接口 | YOLOv8 | YOLOv8 | YOLO11 单类别 DFL |

本项目没有完整照搬 DY-CTMH，因为其层注意力、分类空间图和 DCNv2
会同时引入多个变量，无法判断共享头本身是否有效；此前 FaPN 中的
DCNv2 也已经增加了环境和训练风险。第一版只保留两篇论文共同支持的
“跨尺度共享卷积 + GN”，再加入显式尺度校准解决共享参数可能抹平
P3/P4/P5 分布差异的问题。

## 5. 为什么这样设计

- **针对已有失败经验**：FaPN、FSM、SCG、SGTA 和 P2 路线均未超过
  InceptionDW，因此不再改变 Neck 或检测尺度。
- **适合小批量训练**：GN 不依赖 batch 统计，Colab 的 batch=8 下比
  新增 BN 共享层更稳定。
- **避免过度共享**：不同尺度只共享中间的两层卷积；输入适配、尺度
  系数和输出投影仍然独立。
- **保持定位接口不变**：继续使用 YOLO11 的 16-bin DFL，不再叠加
  新的宽度损失或辅助监督。
- **计算量更低**：640 输入下预审计约为 2.270M 参数、6.015 GFLOPs，
  低于此前 InceptionDW 模型约 2.696M 参数、7.8 GFLOPs。

## 6. 权重继承说明

初始化工具分两层报告：

1. 从 `yolo11n.pt` 到新结构：按名称和形状继承骨干、Neck 与 DFL，
   并把原生 Detect 的三个回归输出卷积映射到新头；
2. 从生成的同构初始化检查点到训练模型：Ultralytics 应显示完整载入。

由于 COCO 权重有 80 个分类输出，而本实验只有一个 `ship` 类别，
三个分类输出层必须随机初始化。两层共享卷积、输入适配和尺度参数也是
新模块，不能伪装成已继承张量。审计 JSON 会记录全部随机初始化键。

## 7. 训练与止损

- 总调度：150 epoch；
- 第一次暂停：80 epoch；
- 初始化：官方 `yolo11n.pt`；
- imgsz：640；
- batch：8；
- optimizer：auto；
- seed：0；
- 第 80 轮保存 `weights/stage80_resume.pt`。

80 轮结果需要与已有 InceptionDW 前 80 轮曲线人工比较。参考值为：

- best mAP50-95：约 0.30168；
- 最后 20 轮 mAP50-95 均值：约 0.27472。

建议继续训练的最低参考为 best mAP50-95 不低于 0.305、最后 20 轮
均值不低于 0.277，且 mAP50 没有明显下降。训练笔记本只展示结果，
不会用硬编码替代最终判断。

## 8. 文件索引

- 模块：`custom_modules/scshared_head.py`
- YAML：`experiments/yolo11n_inceptiondw_scshared_head.yaml`
- 协议：`configs/scshared_head_protocol.yaml`
- 审计：`tools/check_scshared_head.py`
- 训练：`tools/train_scshared_head.py`
- Colab：`notebooks/YOLO11n_InceptionDW_SCSharedHead.ipynb`
- 测试：`tests/test_scshared_head.py`
