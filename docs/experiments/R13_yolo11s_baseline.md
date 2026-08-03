# R13: YOLO11s baseline

## Purpose

R13 measures the effect of increasing official YOLO11 capacity on the frozen
primary ship dataset. It is a model-scale baseline and does not contain DPLS,
CA-SCAM, VGUP, or any other custom module.

## Structure and initialization

- Model: official Ultralytics YOLO11 topology with `scale: s`.
- Detect levels: P3/P4/P5, strides 8/16/32.
- Initialization: independent transfer from `yolo11s.pt`.
- Previous experiment checkpoints: prohibited.
- Dataset split: the same frozen train/validation/test split as R00–R12.

## Frozen training protocol

- Ultralytics 8.4.92, seed 0.
- 150 epochs, `imgsz=640`, batch 8, workers 2.
- Optimizer `auto`; all learning-rate, loss, and augmentation settings are the
  shared values in `experiments/formal_training_config.yaml`.
- Validation selects `best.pt` by maximum validation mAP50–95; the test split
  remains excluded from checkpoint selection.

## Required record

The completed run must retain official-weight Loaded/Total tensors, `args.yaml`,
`results.csv`, `best.pt`, Precision, Recall, mAP50, mAP75, mAP50–95, parameters,
GFLOPs, environment metadata, and checksum-verified Drive artifacts.
