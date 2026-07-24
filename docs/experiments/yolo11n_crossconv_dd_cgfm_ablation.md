# YOLO11n CrossConv / DD / CGFM 实验记录

## 实验状态

- 结构实现：完成
- 官方 `yolo11n.pt` 精确同名同形权重继承审计：完成
- 640×640 CPU 前向：完成
- 正式 Colab GPU 训练：未启动
- 最终指标：待正式训练后回填

## 正式实验记录

| 实验名称 | 结构变化 | 预训练 | Loaded/Total | epoch | imgsz | batch | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo11n-c3cross | Backbone P2/P3/P4/P5 C3k2 使用 CrossConv 单元 | 是 | 479/499 | 150 | 640 | 8 | 待测 | 待测 | 待测 | 待测 |
| yolo11n-dd | Backbone P2/P3/P4/P5 下采样使用 DD | 是 | 475/567 | 150 | 640 | 8 | 待测 | 待测 | 待测 | 待测 |
| yolo11n-cgfm | 第一处 Top-down 融合使用 CGFM | 是 | 498/513 | 150 | 640 | 8 | 待测 | 待测 | 待测 | 待测 |
| yolo11n-inceptiondw-dd | 既有 InceptionDW P2/P3 + DD | 是 | 473/579 | 150 | 640 | 8 | 待测 | 待测 | 待测 | 待测 |
| yolo11n-inceptiondw-cgfm | 既有 InceptionDW P2/P3 + CGFM | 是 | 496/525 | 150 | 640 | 8 | 待测 | 待测 | 待测 | 待测 |

## 诊断对照

`yolo11n-alignconcat-control` 已通过构建和 CPU 前向，但不在首批正式训练列表。它只测试通道对齐加普通拼接，用于解释 CGFM 收益来源。

## 回填规则

80 轮暂停数据只作为人工筛选依据。只有按同一协议完成 150 轮并正式验证后，才回填最终 Precision、Recall、mAP50 和 mAP50-95；同时保留对应 `args.yaml`、`results.csv` 和 commit。
