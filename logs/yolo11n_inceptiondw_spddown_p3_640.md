# Experiment Record

Experiment name: `yolo11n_inceptiondw_spddown_p3_640`

Modification details: Keep the validated InceptionDW P2/P3 C3k2 backbone, native YOLO11 Neck, and Detect(P3,P4,P5). Replace only backbone layer 3 C2-to-C3 stride-2 `Conv` with `SPDDown`.

Pretrained weights: official `yolo11n.pt`

Loaded/Total tensors: `497/511` state tensors in the nc=80 initialization audit. The layer-3 stride-2 kernel and BatchNorm are semantically mapped; InceptionDW replacement kernels remain intentionally random.

Epoch: 150 planned; not started.

imgsz: 640

batch size: 8

Final Precision: pending

Recall: pending

mAP50: pending

mAP50-95: pending

Pre-training status:

- YAML single-variable diff: passed
- Detect strides `[8,16,32]`: passed
- exact SPD pretrained functional mapping: passed
- CPU forward: passed
- 1-epoch tiny-manifest smoke: passed; not a metric experiment
