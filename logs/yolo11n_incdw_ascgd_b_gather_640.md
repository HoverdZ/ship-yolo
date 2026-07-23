# Experiment Record

Experiment Name: `yolo11n_incdw_ascgd_gather_640`

Date: 2026-07-23

Baseline: validated YOLO11n-InceptionDW backbone.

Modification: ASCGD ablation B; one P4 gather/dual-aggregation center with
direct gated residual distribution to P3, P4, and P5. No distribution
cross-attention.

Pretrained Weights: official `yolo11n.pt`

Loaded/Total State Tensors: 308/538

Loaded/Total Parameter Elements: 1,746,544/2,701,837

Training:

- Epochs: 150 planned; not started
- imgsz: 640
- batch: 8
- workers: 2
- seed: 0

Results:

- Precision: not available
- Recall: not available
- mAP50: not available
- mAP50-95: not available

Status: CPU build/forward passed. Formal training was intentionally not
started.
