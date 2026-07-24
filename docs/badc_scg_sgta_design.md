# BADC-SCG-SGTA small-ship experiment family

## Decision

The existing 150-epoch InceptionDW run is the matched reference. Its first 80
epochs are reused; the baseline is not retrained. Three independent 80-epoch
screens are prepared:

1. `badc`: replace shallow P2/P3 InceptionDW blocks with BADC.
2. `scg`: retain InceptionDW and replace only the P3 concat with SCG.
3. `sgta`: retain the exact InceptionDW architecture and change only training
   assignment/localization quality.

The complete `full` model is implemented and audited but intentionally excluded
from the default Colab screening choices until the independent results justify
combination.

## BADC: background-aware directional contrast

BADC estimates low-frequency background with a 7x7 average filter and applies
3x3, 1x11, and 11x1 depthwise branches to the residual contrast. A two-channel
spatial gate generated from background magnitude and contrast magnitude selects
the three branches with a per-position softmax. The transformed residual uses a
zero-initialized scalar, so each replacement bottleneck begins as an identity
residual and learns the directional contrast gradually.

BADC is confined to the same two P2/P3 C3k2 positions as the matched
InceptionDW experiment.

## SCG: bounded semantic confirmation

SCG replaces the original P3 `Concat` layer. It uses the upsampled P4 semantic
feature and the backbone C3 feature to produce a single spatial mask, then
modulates C3 within the fixed interval `[0.75, 1.25]`.

The final mask convolution is zero-initialized. Therefore the initial output is
exactly `Concat(upsampled_P4, C3)`. Replacing the concat in place preserves
every downstream layer index and pretrained parameter key. This design fixed an
early audit failure where insertion as an extra layer reduced transfer from
497/525 to 262/525 state tensors.

## SGTA: scale-adaptive Gaussian task alignment

SGTA does not change inference architecture. During training, it defines:

`quality = lambda(size) * CIoU + (1 - lambda(size)) * NWD`

where `lambda(size)` is a smooth sigmoid centered at 32 pixels with temperature
6 pixels. NWD uses normalization constant 12.8. The same quality is used by the
TaskAlignedAssigner and box regression loss. Native DFL remains unchanged.

Small boxes therefore receive stable Gaussian geometry while normal boxes
converge to the original IoU behavior. SGTA has zero inference parameters and
zero inference FLOPs.

## Staged 80 + 70 protocol

The first stage is not an independent 80-epoch schedule. Training is configured
for a 150-epoch scheduler from the beginning, matching the existing InceptionDW
run, and is stopped by a callback after epoch 80.

Ultralytics strips optimizer state from `last.pt` when a run ends normally. The
callback therefore copies the raw epoch-80 checkpoint to
`weights/stage80_resume.pt` before final checkpoint stripping. Only this
checkpoint is used to continue from epoch 81 to the total of 150 epochs.

This preserves:

- model and EMA weights;
- optimizer and AMP scaler;
- learning-rate scheduler horizon;
- epoch counter and original training arguments.

Starting a new 70-epoch training job from `best.pt` or `last.pt` is not an
equivalent continuation and is rejected by the training entrypoint.

## Preflight summary

At 640 input and `nc=80` parser audit:

| Variant | Parameters | GFLOPs | Loaded state tensors |
| --- | ---: | ---: | ---: |
| BADC | 2,620,564 | 6.5510 | 487/511 |
| InceptionDW + SCG | 2,627,773 | 6.6261 | 497/525 |
| InceptionDW + SGTA | 2,619,164 | 6.5142 | 497/511 |
| Full | 2,629,173 | 6.6628 | 487/525 |

All variants retain Detect(P3, P4, P5) with strides 8, 16, and 32. Reports are
stored in `reports/badc_scg_sgta/`.

## Training entrypoints

- Preflight: `python tools/check_badc_scg_sgta.py --weights yolo11n.pt`
- Direct training: `python tools/train_badc_scg_sgta.py --help`
- Colab generator: `python tools/build_badc_scg_sgta_colab.py`

Formal training must run directly in the notebook kernel so live Ultralytics
output remains visible.
