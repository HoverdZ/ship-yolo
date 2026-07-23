# Experiment Record

Experiment Name: `yolo11n_incdw_ascgd_symmetric_640`

Date: 2026-07-23

Baseline: validated YOLO11n-InceptionDW backbone.

Modification: ASCGD ablation G; every P3/P4/P5 distribution path computes and
fuses spatial and channel cross-attention without adding another aggregation
block.

Pretrained Weights: official `yolo11n.pt`

Loaded/Total State Tensors: 308/615

Loaded/Total Parameter Elements: 1,746,544/3,418,767

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
started. Relative to E, G adds 402,514 parameters and 1.661039 reported THOP
GFLOPs at 640.
