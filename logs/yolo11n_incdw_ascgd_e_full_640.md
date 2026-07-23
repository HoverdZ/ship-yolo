# Experiment Record

Experiment Name: `yolo11n_incdw_ascgd_full_640`

Date: 2026-07-23

Baseline: validated YOLO11n-InceptionDW backbone.

Modification: full ASCGD-Neck (E); one gather/dual-aggregation center,
high-to-low P3 window spatial cross-attention, and low-to-high P4/P5
cross-covariance channel attention.

Pretrained Weights: official `yolo11n.pt`

Loaded/Total State Tensors: 308/574

Loaded/Total Parameter Elements: 1,746,544/3,016,253

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

Status: CPU batch-2 640 forward, rectangular forward, backward, finite-output,
and complete-gradient checks passed. This is the first formal Colab training
candidate; training was intentionally not started locally.
