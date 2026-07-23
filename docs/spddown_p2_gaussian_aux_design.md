# YOLO11n-InceptionDW: SPDDown and Training-Only P2 Gaussian Supervision

Status: implementation, initialization, CPU audits, and tiny-manifest smoke
training complete. No formal dataset training has been started.

## Motivation from the ship dataset

The inspected `lw2.zip` copy contains 688 validation instances. At an input
size of 640, approximately 84.9% have area below `32 x 32`, while the median
minimum side is about 21.3 pixels. The main risk is therefore losing weak
spatial evidence before P3 or suppressing it during feature selection.

Previous FaPN, FSM, spatial-gating, and ASCGD-style Neck experiments produced
negative or precision-heavy/recall-poor behavior. These experiments restore
the native YOLO11 Neck and change one variable at a time.

Primary references:

- SPD-Conv: <https://arxiv.org/abs/2208.03641>
- authors' official MIT-licensed code: <https://github.com/LabSAINT/SPD-Conv>
- RFLA: <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3138_ECCV_2022_paper.php>
- TTFNet Gaussian encoding: <https://ojs.aaai.org/index.php/AAAI/article/view/6838>

The implementation is an experiment-specific adaptation, not a claim of
bit-for-bit reproduction of an entire detector from those papers.

`custom_modules/spd.py` is not a verbatim copy of the authors' YOLOv5 code.
It is a YOLO11/Ultralytics-compatible adaptation. Its four-phase
space-to-depth order exactly matches the official `space_to_depth` operation,
then the wrapper applies a non-strided Ultralytics `Conv`. It additionally
provides input validation and deterministic pretrained-weight mapping needed
by this repository.

## Experiment A: targeted SPDDown

YAML:
`experiments/yolo11n_inceptiondw_spddown_p3.yaml`

Only backbone layer 3 changes:

```text
InceptionDW C2 feature (stride 4)
  -> four phase space-to-depth rearrangement
  -> 3x3 stride-1 Conv + BN + SiLU
  -> C3 feature (stride 8)
```

All pixels are rearranged into channels before convolution. P4/P5, the
complete FPN/PAN, and Detect(P3,P4,P5) are unchanged.

### Exact pretrained initialization

A stride-2 3x3 convolution can be represented by space-to-depth followed by a
stride-1 convolution. `inflate_stride2_conv_to_spd()` places the original nine
kernel offsets into the matching phase channel and coarse-grid offset. Unused
weights are initialized to zero. BatchNorm is copied directly.

The CPU equivalence audit reports a maximum absolute difference around
`5.3e-6`, caused by floating-point accumulation order. The transformed layer
therefore starts from the official downsampler's function instead of a random
replacement.

The nc=80 initialization audit loads 497/511 state tensors and initializes
2,728,912/2,729,756 target parameter elements from official weights or the
exact SPD mapping.

## Experiment B: training-only P2 Gaussian supervision

YAML:
`experiments/yolo11n_inceptiondw_p2_gaussian_aux.yaml`

The native detector still consumes exactly P3, P4, and P5. During training,
layer-2 P2 is also passed to a one-channel 1x1 auxiliary head. Every ground
truth box creates a max-composed Gaussian heatmap at stride 4:

```text
sigma = clamp(0.25 * sqrt(box_width_P2 * box_height_P2), 1, 3)
```

The dense auxiliary loss is:

```text
L_aux = L_positive_soft_focal + L_background_focal
L_total = L_box + L_cls + L_dfl + 0.25 * L_aux
```

Soft Gaussian neighborhoods supply more shallow supervision than a single
center pixel. Background terms remain present but are focal-weighted.

In evaluation and export mode the auxiliary convolution is not executed.
Inference output, detection strides, and the number of detection heads remain
native YOLO11 values. The auxiliary parameters remain in a training
checkpoint, but contribute zero inference operations.

The nc=80 initialization audit loads 497/513 state tensors and
2,618,320/2,619,229 target parameter elements. The only additional random
parameters are the auxiliary 1x1 head, alongside the already intentional
InceptionDW replacements.

## Registration and training integration

`custom_modules/register.py` registers both modules without editing
`site-packages`. The parser patch is idempotent and pinned to
`ultralytics==8.4.92`.

`tools/train_spddown_p2aux.py` exposes `run_training(TrainingRequest(...))`.
The Colab notebook imports and calls this function directly in the notebook
kernel. Formal training is not launched through `subprocess`, `Popen`, `!python`,
or a background shell, so Ultralytics progress remains visible.

The P2 experiment uses a small `DetectionTrainer` subclass only to construct
`P2GaussianDetectionModel` and name the fourth loss item. The official
Ultralytics optimizer, scheduler, AMP, dataloaders, validator, callbacks,
checkpointing, early stopping, and progress loop remain in use.

## Dataset and Colab I/O

The Drive source configured in the notebook is:

```text
/content/drive/MyDrive/ship_detection/data
```

Before training, every file is copied to:

```text
/content/datasets/ship_detection
```

The copy cell uses `ThreadPoolExecutor` plus `shutil.copyfile`. It does not
train from mounted Drive. A new local `data.yaml` is generated after detecting
either `train/images` or `images/train` layout.

The cloud dataset at the configured Drive path is authoritative. The notebook
prints the discovered split counts for provenance, but does not compare them
with local or historical fixed counts and does not reject a valid cloud copy
because its counts differ.

## Required order

1. Run the notebook environment and repository cells.
2. Copy and audit the dataset.
3. Run both 640 preflight audits.
4. Train `spddown` alone.
5. Train `p2_gaussian_aux` alone.
6. Compare each with the matched InceptionDW baseline.
7. Do not combine the two unless both individual experiments are positive.

At epoch 30, stop a candidate when its best Recall trails the matched baseline
by more than 0.02 and its best mAP50-95 trails by more than 0.005. This is a
resource gate, not a replacement for the 150-epoch formal result.

## Verified checks

- both YAML files build with `ultralytics==8.4.92`;
- each YAML has exactly one intended difference from the InceptionDW baseline;
- model strides stay `[8,16,32]`;
- SPD semantic initialization is functionally equivalent;
- P2 auxiliary loss is positive and finite;
- auxiliary and P2 backbone gradients are finite and nonzero;
- evaluation does not execute the P2 auxiliary convolution;
- official weight initialization saves and reloads;
- both variants complete a direct-process one-epoch tiny-manifest CPU smoke;
- no formal training metrics have been fabricated or inferred from smoke runs.
