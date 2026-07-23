# Experiment Record

Experiment name: `yolo11n_inceptiondw_p2_gaussian_aux_640`

Modification details: Keep the validated InceptionDW backbone, native YOLO11 Neck, and Detect(P3,P4,P5). Add a P2 one-channel Gaussian heatmap branch and weighted auxiliary loss only while training.

Pretrained weights: official `yolo11n.pt`

Loaded/Total tensors: `497/513` state tensors in the nc=80 initialization audit. Native Detect tensors inherit by exact name and shape; the P2 auxiliary convolution and InceptionDW replacement kernels remain intentionally random.

Epoch: 150 planned; not started.

imgsz: 640

batch size: 8

Final Precision: pending

Recall: pending

mAP50: pending

mAP50-95: pending

Pre-training status:

- YAML single-variable diff: passed
- native Detect strides `[8,16,32]`: passed
- auxiliary loss and P2 gradient: passed
- auxiliary branch not called during evaluation: passed
- CPU forward: passed
- 1-epoch tiny-manifest smoke with visible `p2_aux_loss`: passed; not a metric experiment
