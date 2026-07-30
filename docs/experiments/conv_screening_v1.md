# YOLO11n 浅层卷积筛选 v1

## 目标与实验边界

本轮回答一个单独问题：在 YOLO11n 的浅层 Backbone 中，哪类空间卷积更适合当前光学遥感小船数据？所有实验都只替换 P2/P3 后两个 `C3k2` 内部 `Bottleneck.cv2`；`Bottleneck.cv1`、下采样卷积、P4/P5、Neck、Detect Head、损失和训练协议均保持不变。

需要特别区分：

- 本实验的 PConv 是 **Pinwheel-shaped Convolution（风车形卷积）**，不是 FasterNet 的 Partial Convolution。
- 每个实验只使用论文中的卷积算子，不同时加入论文配套损失、注意力、FFN 或检测头。
- 训练前从官方 `yolo11n.pt` 按“同名且同形状”继承张量。审计必须证明 P2/P3 的 `Bottleneck.cv1` 全部继承成功，未继承范围只能来自新 `cv2` 和 COCO 80 类到船舶 1 类所必然重建的分类分支。

## 数据驱动的选择依据

项目已有的确定性尺度分析以 `imgsz=640` 为目标尺度：训练集船舶短边 Q5、Q10 和中位数约为 12.50、13.75 和 21.25 px；约 17.01% 的训练目标短边小于 16 px，约 91.84% 小于 32 px。水平框长宽比中位数约 1.25，只有约 3.15% 大于等于 2。

因此，首轮优先检验三种互补假设：

1. **方向敏感的紧凑局部采样**，避免极小目标被普通方形卷积和背景共同稀释；
2. **局部/大范围上下文的动态选择**，帮助区分小船与浪花、岸线和高光；
3. **密集多尺度感受野**，覆盖当前数据中从约十余像素到数十像素的尺度变化。

当前数据并不支持把“普遍超大长宽比”作为首要动机，因此 PConv 的论文表述应聚焦小目标方向结构，而不是夸大船体细长性。

## 入选实验

| ID | 唯一结构变化 | 原始论文与层次 | 官方实现 | 面向的问题 | 对当前任务的假设 |
|---|---|---|---|---|---|
| C1 | P2/P3 `Bottleneck.cv2` → Pinwheel PConv | *Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection*, AAAI 2025，人工智能顶级会议 | `JN-Yang/PConv-SDloss-Data`，commit `a801f043c83f73aa9af9ab2f689e59ebef928fc4`，MIT | 普通方形采样难以突出极小目标的中心—方向局部模式 | 四向非对称填充与 `1×k/k×1` 分支可增强弱小船局部结构，同时保持输出分辨率 |
| C2 | P2/P3 `Bottleneck.cv2` → LSKConv | *Large Selective Kernel Network for Remote Sensing Object Detection*, ICCV 2023，计算机视觉顶级会议 | `zcablii/LSKNet`，commit `386cbefc71d402e7a9375495bbe34d5c2aec0e37`，CC BY-NC 4.0 | 遥感目标需要随目标和背景动态选择局部或长程上下文 | 小船本体像素少，选择性大核可利用水域上下文抑制岸线、浪花和高光误检 |
| C3 | P2/P3 `Bottleneck.cv2` → PKIConv | *Poly Kernel Inception Network for Remote Sensing Detection*, CVPR 2024，计算机视觉顶级会议 | `PKINet/PKINet`，commit `a33aa22d188c9946cc83fba60e3bb8ac0ec82ff7`，Apache-2.0 | 遥感目标尺度变化大，单一卷积核难以兼顾不同感受野 | `3/5/7/9/11` 密集、非空洞深度卷积并行响应可覆盖当前小船尺度分布，并减少空洞采样漏掉弱特征的风险 |

论文与代码入口：

- PConv：[AAAI 2025 论文](https://ojs.aaai.org/index.php/AAAI/article/view/32996)，[官方代码](https://github.com/JN-Yang/PConv-SDloss-Data)
- LSKNet：[ICCV 2023 论文](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.html)，[官方代码](https://github.com/zcablii/LSKNet)
- PKINet：[CVPR 2024 论文](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)，[官方代码](https://github.com/PKINet/PKINet)

## 遥感船舶论文交叉核验

本轮没有只依赖通用小目标论文，还对光学、SAR 和红外船舶检测工作进行了交叉核验：

- *EPFNet: Lightweight feature fusion for small ship detection in remote sensing imagery*（Digital Signal Processing, 2025）明确把复杂背景下的细粒度特征不足、噪声干扰和跨尺度融合列为小船漏检/误检来源。这支持将浅层细节、背景抑制和尺度响应作为本轮三个主要假设。
- SAR-NanoShipNet（ISPRS Journal of Photogrammetry and Remote Sensing, 2025）使用面向小船的可变形/边界感知特征提取。该刊按 SJR 2024 为 Q1，但其 DABConv 同时混合可变形卷积和边界注意力，且针对 SAR 成像；当前数据为光学图像，前期 DCNv2 又显示出明显速度代价，所以未作为纯卷积首轮实验。
- IRSD-DETR（ISPRS Journal of Photogrammetry and Remote Sensing, 2026）将 Pinwheel-shaped Convolution 用于红外小船检测，为 C1 提供了比通用红外小目标更直接的船舶证据；但光学与红外仍有域差异，必须由当前数据实证。
- 近期光学船舶工作也反复采用大核或移位大核扩大上下文，这与 LSK/PKI 的方向一致；本轮选择已有顶会、官方实现和遥感基准验证的 LSK/PKI，避免同时引入完整检测器的其他变量。

参考入口：

- [EPFNet](https://www.sciencedirect.com/science/article/pii/S1051200425001745)
- [SAR-NanoShipNet](https://www.sciencedirect.com/science/article/pii/S0924271625004903)
- [IRSD-DETR](https://www.sciencedirect.com/science/article/abs/pii/S0924271626001681)
- [ISPRS Journal 的 SJR 2024 Q1 记录](https://www.scimagojr.com/journalsearch.php?q=29161&tip=sid)

“一区/二区”会随年份、数据库和学科分类变化。本文档只把可核验的 SJR 年份写死，不把当前分区反向套用到所有历史年份。

## 未进入首轮的候选

- **DCNv2**：CVPR 2019，几何自适应能力强，但项目已实际观察到明显训练减速；重复实验的信息增益低。
- **DABConv**：船舶任务证据直接，但同时包含可变形卷积与边界注意力，无法形成“只换一种卷积”的干净消融，且存在光学/SAR 模态差异。
- **Switchable Atrous Convolution / DetectoRS**：CVPR 2021，尺度自适应明确；但稀疏空洞采样对十余像素弱目标存在欠采样风险。PKINet 的密集非空洞多核更适合作为当前首轮。
- **ODConv**：ICLR 2022，动态多维卷积有通用价值，但对遥感小船的直接证据弱于本轮三项，动态权重也增加训练复杂度。
- **Involution**：CVPR 2021，在通用检测中有效，但并非卷积且缺少当前任务所需的直接小目标/船舶证据。
- **SPDConv**：会改变下采样语义，不是对 stride-1 `Bottleneck.cv2` 的等位替换，无法保持单变量。

## 官方实现适配说明

- `PinwheelConv` 保留官方四组非对称 Padding、共享横/纵卷积和 `2×2` 融合；不使用 SD Loss。
- `LargeSelectiveKernelConv2d` 保留官方 `5×5 DW → 7×7 DW(dilation=3)`、平均/最大空间统计和选择门控；仅在输出后补充 BN+SiLU，以匹配被替换 YOLO `cv2` 的输出约定。
- `PolyKernelConv2d` 提取官方 PKI Inception Bottleneck 中 `3/5/7/9/11`、dilation 全为 1 的卷积混合器；不加入 CAA、ConvFFN、DropPath 或外层 PKIBlock。这样 C3 的解释仍然是“密集多尺度卷积”，而不是完整 PKINet。

这些代码是面向当前私有、非商业科研仓库的适配。后续若公开发布或商业使用，必须重新核查第三方许可证，尤其是 LSKNet 的 CC BY-NC 4.0 条款。

## 固定训练协议

- Ultralytics: `8.4.92`
- 初始化：官方 `yolo11n.pt`，同名同形状张量继承；Notebook 中 `pretrained=True` 仅用于让 8.4.92 将已初始化内存模型交给 Trainer，不会再次替换成另一套权重
- epoch: 150
- imgsz: 640
- batch: 8
- workers: 2
- seed: 0
- cache: `disk`
- deterministic: `False`
- save_period: 10
- 训练入口：Notebook 当前进程直接调用官方 `YOLO.train(...)`，严禁训练子进程
- 数据：Drive 数据集通过 16 线程 `shutil.copyfile` 复制到 `/content/ship_detection/data`，显示文件数和已处理字节双进度；不比较固定文件数量
- 验证集只用于模型选择；测试集默认封存

## 实验记录

| 字段 | C1 PConv | C2 LSKConv | C3 PKIConv |
|---|---:|---:|---:|
| 实验名称 | `C1_pconv_p23` | `C2_lskconv_p23` | `C3_pkiconv_p23` |
| 结构变化 | P2/P3 cv2 only | P2/P3 cv2 only | P2/P3 cv2 only |
| 使用预训练权重 | 是，官方 `yolo11n.pt` | 是，官方 `yolo11n.pt` | 是，官方 `yolo11n.pt` |
| Loaded/Total tensors | 运行前自动记录 | 运行前自动记录 | 运行前自动记录 |
| epoch / imgsz / batch | 150 / 640 / 8 | 150 / 640 / 8 | 150 / 640 / 8 |
| Precision | 待训练 | 待训练 | 待训练 |
| Recall | 待训练 | 待训练 | 待训练 |
| mAP50 | 待训练 | 待训练 | 待训练 |
| mAP50-95 | 待训练 | 待训练 | 待训练 |

本轮不预设三个卷积必然优于基线。只有在同一数据、初始化和训练协议下取得可重复提升，才可进入论文正式结构；负结果同样用于说明小目标卷积适配的边界。
