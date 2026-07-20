# Experiment Record

Experiment Name: YOLO11n-SCConv-C3k2-Full
Date: 2026-07-20

Baseline: Official Ultralytics 8.4.92 YOLO11n (`yolo11n.yaml`, `yolo11n.pt`)

Modification: Replace every internal 3×3 stride=1 Conv2d in the four
Backbone C3k2 nodes with SCConv while preserving outer CSP/C2f topology,
1×1 Conv, downsampling Conv, SPPF, C2PSA, Neck, and Detect(P3/P4/P5).

Weight Transfer: Official `yolo11n.pt`, name-and-shape matching
Loaded/Total: 487/571 state tensors; 98.362993% of parameter elements

Training: Not started locally; reserved for Colab/GPU
Epoch: 150 planned
imgsz: 640 planned
batch: 8 planned

Results: Not available; formal training has not started.
Precision: pending
Recall: pending
mAP50: pending
mAP50-95: pending

Analysis: CPU unit, C3k2 alignment, model build, 640×640 forward, topology,
GFLOPs, and weight-transfer checks passed. New unmatched parameters are
confined to the four Backbone SCConv nodes; Neck and Detect transfer fully.

Conclusion: Code is ready for a controlled Colab training run. No accuracy or
latency claim is made before formal training and GPU measurement.

Next Step: Check out `experiment/scconv-c3k2-full`, rerun preflight checks in
Colab, then start the 150-epoch single-class ship experiment.
