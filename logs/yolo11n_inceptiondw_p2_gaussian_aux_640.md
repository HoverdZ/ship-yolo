# Experiment Record

Experiment name: `yolo11n_inceptiondw_p2_gaussian_aux_640`

Modification details: Keep the validated InceptionDW backbone, native YOLO11 Neck, and Detect(P3,P4,P5). Add a P2 one-channel Gaussian heatmap branch and weighted auxiliary loss only while training.

Pretrained weights: official `yolo11n.pt`

Loaded/Total tensors: `497/513` state tensors in the nc=80 initialization audit. Native Detect tensors inherit by exact name and shape; the P2 auxiliary convolution and InceptionDW replacement kernels remain intentionally random.

Epoch: 150 planned. The first formal attempt is invalid and must not be resumed.

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

Numerical incident and correction:

- invalid run: epoch 1 `train/p2_aux_loss=inf`; epochs 2 onward `nan`
- root cause: FP16 sigmoid saturation followed by `log(1-p)`
- correction: auxiliary focal calculation promoted to FP32 and rewritten with
  stable `F.logsigmoid` identities
- regression coverage: saturated logits from -100 to 100 in FP16/FP32, finite
  forward and gradients, plus low-precision agreement with FP32
- restart policy: use a new run directory; never resume the invalid checkpoint
