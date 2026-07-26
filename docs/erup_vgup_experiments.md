# ERUP and VGUP image-adaptive YOLO11n experiments

## Scope and source status

The source method is *ERUP-YOLO: Enhancing Object Detection Robustness for
Adverse Weather Condition by Unified Image-Adaptive Processing*, by Yuka
Ogino, Yuho Shoji, Takahiro Toizumi, and Atsushi Ito, WACV 2025. The primary
references are the
[CVF paper](https://openaccess.thecvf.com/content/WACV2025/html/Ogino_ERUP-YOLO_Enhancing_Object_Detection_Robustness_for_Adverse_Weather_Condition_by_WACV_2025_paper.html)
and its
[supplementary material](https://openaccess.thecvf.com/content/WACV2025/supplemental/Ogino_ERUP-YOLO_Enhancing_Object_WACV_2025_supplemental.pdf).

No author-maintained official code repository was found during this
implementation. `custom_modules/erup.py` is therefore a paper-level
reproduction from the published equations and architecture, not a line-by-line
official-code port. BPW and KBL are not claimed as project-original work.

This change prepares exactly four configurations and does not launch formal
training. It adds no fog labels, fog augmentation, restoration loss,
consistency loss, or ablation-only model.

## Reproduced ERUP path

`ERUPPreprocessor` runs before the detector's first Conv:

```text
RGB input -> wide parameter encoder -> BPW -> KBL -> existing detector
```

- The BPW filter uses the paper's cubic Bezier curve with endpoints `(0,0)`
  and `(1,1)`, two polar-parameterized control points, four parameters per RGB
  channel, and 12 parameters total. Equation 12 is evaluated by a
  differentiable piecewise mapping. Zero parameters produce the identity
  curve.
- KBL uses the exact published form
  `I * Conv(I, K1) + Conv(I, K2) + I`. The two per-image, per-RGB 9x9 kernels
  contain `2 * 3 * 9 * 9 = 486` parameters. Reshaped grouped convolution keeps
  samples and channels independent.
- The encoder keeps five ordinary 3x3 stride-1 convolutions with channels
  `3 -> 64 -> 128 -> 256 -> 512 -> 1024`. Each is followed by ReLU and 3x3
  stride-2 average pooling, then global average pooling, FC(498), sigmoid, and
  mapping to `[-1,1]`.
- ERUP has no global or spatial gate. Its 498 outputs split into BPW 12 and KBL
  486 parameters.

The paper fixes the equations, 9x9 kernels, channels, pooling pattern, and
filter order. It does not publish an official implementation, an explicit BPW
segment count, all activation details, or parameter-head initialization. This
reproduction uses eight BPW segments, ReLU after encoder convolutions, and a
small normal FC initialization with zero bias. Eight is the only explicit
piecewise-filter segment count in the same method section and is recorded as
an engineering recovery, not a confirmed hidden implementation detail.

## Complete VGUP path

VGUP retains the same tested `BPWFilter` and `KBLFilter`; it is one complete
module with three inseparable modifications:

1. A global BPW residual acceptance gate:
   `I_bpw = I + g_b * (BPW(I) - I)`, where `g_b` is `[B,1,1,1]`.
2. A KBL spatial visibility gate:
   `I_out = I_bpw + M * (KBL(I_bpw) - I_bpw)`, where `M` is restored to
   `[B,1,H,W]` by bilinear interpolation and is shared by RGB channels.
3. A lightweight shared encoder. The image is resized only for parameter
   prediction to 128x128, then processed by a 3-to-16 Conv stem and
   depthwise-separable stages `16 -> 32 -> 64 -> 128`. One feature tensor
   predicts the low-resolution spatial gate; GAP predicts 498 filter
   parameters and the global gate.

Both gates use sigmoid. The initial global-gate bias is `-1.5` and the spatial
gate bias is `-1.0`, so training starts conservatively without hard-coding
either branch off. Gate supervision is exclusively the detector loss.

These three changes always appear together because the experiment tests the
claimed VGUP design, not three unplanned ablations. The wide ERUP encoder is
not used by VGUP, and VGUP does not inherit ERUP weights.

## Base topology and naming audit

The repository contains no independent module or successful YAML literally
named `SFL`. Exhaustive local and remote branch inspection found the two
successful structures at:

- `experiments/yolo11n_inceptiondw_dysample_pls.yaml`
- `experiments/yolo11n_inceptiondw_dysample_pls_scam.yaml`

Their PLS topology is P2/P3/P4 detection with strides 4/8/16. The task's SFL
label is retained in the four requested filenames and experiment names, while
the detector graph is derived byte-for-byte in layer order and arguments from
these real PLS configurations. No new meaning or new implementation of SFL was
invented. After inserting layer 0, every base layer type matches at offset +1.
InceptionDW, DySample, PLS, SCAM, Detect, DFL, channels, and positions are
unchanged.

## Four controlled configurations

| Experiment | YAML | Base model | SCAM | Preprocessor | Strides |
| --- | --- | --- | --- | --- | --- |
| `incdw_dysample_sfl_scam_vgup` | `experiments/yolo11n_incdw_dysample_sfl_scam_vgup.yaml` | PLS + SCAM | yes, 3 | VGUP | 4/8/16 |
| `incdw_dysample_sfl_vgup` | `experiments/yolo11n_incdw_dysample_sfl_vgup.yaml` | PLS | no | VGUP | 4/8/16 |
| `incdw_dysample_sfl_scam_erup` | `experiments/yolo11n_incdw_dysample_sfl_scam_erup.yaml` | PLS + SCAM | yes, 3 | ERUP | 4/8/16 |
| `incdw_dysample_sfl_erup` | `experiments/yolo11n_incdw_dysample_sfl_erup.yaml` | PLS | no | ERUP | 4/8/16 |

The SCAM/no-SCAM pairs differ only in the existing three SCAM layers. The
ERUP/VGUP pair for each base differs only at layer 0.

## CPU statistics

Generated with Ultralytics 8.4.92, input 640, and
`tools/check_erup_vgup_models.py`:

| Experiment | Parameters | Layer-0 parameters | GFLOPs | Official -> base | Base detector -> target |
| --- | ---: | ---: | ---: | ---: | ---: |
| VGUP + SCAM | 1,447,106 | 77,396 | 18.9409 | 126/514 | 514/514 |
| VGUP | 1,372,400 | 77,396 | 18.4067 | 126/457 | 457/457 |
| ERUP + SCAM | 8,150,752 | 6,781,042 | 77.7599 | 126/514 | 514/514 |
| ERUP | 8,076,046 | 6,781,042 | 77.2257 | 126/457 | 457/457 |

THOP counts the model graph but may undercount BPW's piecewise arithmetic and
per-sample dynamic KBL convolution; the GFLOPs values are useful for consistent
repository comparisons, not deployment latency claims.

## Weight inheritance

All four models start independently from the same official `yolo11n.pt`:

1. Build the exact corresponding PLS or PLS+SCAM base with the dataset class
   count.
2. Apply the repository's exact-name/exact-shape official transfer to that
   base and retain the full unmatched-key audit.
3. Copy every base detector tensor into the adaptive target with the expected
   +1 layer offset.
4. Leave every ERUP/VGUP key at its own initialization.
5. Save and reload `initialization.pt`; immediately before training, compare
   every tensor in the Trainer with that audited checkpoint.

Thus 126 tensors originate directly from official YOLO11n in each base, while
the remaining custom detector tensors retain the corresponding independently
initialized base state. The second transfer is 457/457 for no-SCAM and 514/514
for SCAM. `strict=False` is never silent: missing preprocessor keys, unexpected
keys, shape mismatches, and post-load verification failures are all reported.

## Training workflow

The committed Colab notebook is
`colab/YOLO11n_ERUP_VGUP_Final_Models.ipynb`. It pins
`ultralytics==8.4.92`, securely clones the private branch, and copies
`/content/drive/MyDrive/ship_detection/data` to
`/content/ship_detection/data` with 16-thread `shutil.copyfile` plus live file
and byte progress. It does not compare fixed image/label counts.

Formal settings are identical across all four configurations: 150 epochs,
640, batch 8, workers 2, seed 0, optimizer `auto`, patience 150,
`cache="disk"`, and the established loss/augmentation values. No new
augmentation is introduced. `cache="disk"` avoids the nondeterministic RAM
cache warning. `deterministic=False` is required by DySample's CUDA
`grid_sample` backward and prevents warning tracebacks from corrupting the
official progress display.

The training Cell directly calls `train_experiment(...)`, which directly calls
Ultralytics `YOLO.train(...)` in the notebook kernel. Training is never placed
in a subprocess. The committed switch is `RUN_TRAINING=False` because this
round must not start a formal run.

Equivalent command-line preparation is:

```powershell
python tools/train_erup_vgup_models.py `
  --experiment incdw_dysample_sfl_vgup `
  --data C:\path\to\data.yaml `
  --weights yolo11n.pt `
  --name yolo11n_incdw_dysample_sfl_vgup_640
```

## Validation and visualization

```powershell
pytest -q
python tools/check_erup_vgup_models.py --weights yolo11n.pt --imgsz 640
python tools/visualize_erup_vgup.py --image C:\path\to\ship.jpg
git diff --check
git status
```

The audit covers BPW identity and gradients, KBL sample/RGB isolation and both
kernel gradients, batch 1/2, ERUP/VGUP debug shapes, gate ranges and 0/1
boundaries, all four 640 forwards, Detect P2/P3/P4 shapes, minimal backward
through filters/gates/detector, strict checkpoint roundtrip, SCAM presence,
parameter/GFLOPs reporting, and native/existing-model compatibility. Evidence
is in `reports/erup_vgup/erup_vgup_model_check.json`.

## Deliberately unresolved

- No formal training or accuracy metrics were produced.
- ONNX, TensorRT, dynamic-kernel export, deployment latency, and memory tuning
  are not addressed.
- No fog augmentation, fog labels, restoration objectives, or follow-up
  ablation was added.
- The original ERUP paper's unpublished implementation details remain an
  experimental-reproduction risk.
- CPU/THOP success does not replace a Colab GPU AMP smoke run; that is the
  first check before committing 150 epochs.
