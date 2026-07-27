# CA-SCAM experiment

## Scope

This round implements only **CA-SCAM**. PC-SCAM is intentionally deferred by
the experiment owner and has no module, YAML, training entrypoint, or Colab
notebook in this branch.

The controlled baseline is the trained
`yolo11n_incdw_dysample_sfl_scam_vgup` topology:

- VGUP at layer 0;
- InceptionDW in the P2/P3 backbone stages;
- two DySample upsampling blocks;
- PLS/SFL P2/P3/P4 detection topology;
- three independent SCAM blocks directly before Detect.

Its best Drive result occurred at epoch 143:

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.85610 | 0.73692 | 0.80119 | 0.34787 |

## Hypothesis

Small ships can have weak local contrast against sea clutter. CA-SCAM keeps
the original FFCA-YOLO context residual and learns a bounded, spatially
conditioned calibration from local contrast:

```text
L = AvgPool3x3(F)
D = mean_channel(abs(F - L))
A = sigmoid(Conv3x3(D))
beta = max_delta * tanh(contrast_logit)
out = F + (1 + beta * A) * SCAMResidual(F)
```

`max_delta=0.1`. The projection weight, projection bias, and scalar logit are
zero initialized, so `beta=0` and the initial output is bit-exact to the
original SCAM. `count_include_pad=False` keeps local contrast zero for a
constant feature map, including its border.

## Controlled change

The new YAML is
`experiments/yolo11n_incdw_dysample_pls_ca_scam_vgup.yaml`.
Relative to the successful YAML, only layers 22, 23, and 24 change from
`SCAM` to `CASCAM [0.1]`. Detect still consumes `[22, 23, 24]`, and the
strides remain `[4, 8, 16]`.

Each CA-SCAM adds one 3×3 weight tensor, one bias, and one scalar: 11
parameters. Three blocks therefore add exactly 33 parameters.

## Initialization contract

The experiment does **not** initialize from the successful run's `best.pt`.
It follows the same official `yolo11n.pt` initialization route as the
successful model:

1. build the unchanged successful topology;
2. inherit all compatible official YOLO11n detector tensors;
3. copy every same-name/same-shape tensor into CA-SCAM;
4. leave only these nine new state tensors at zero:
   `contrast_logit`, `contrast_proj.weight`, and `contrast_proj.bias` at
   layers 22, 23, and 24.

The expected inheritance audit is `562/571` tensors. All original SCAM
`k/v/m/m2` tensors are included among the 562 copied tensors.

## Formal training protocol

| Setting | Value |
|---|---|
| epochs | 150 |
| imgsz | 640 |
| batch | 8 |
| workers | 2 |
| optimizer | Ultralytics `auto` |
| seed | 0 |
| cache | disk |
| deterministic | false |
| initialization | official `yolo11n.pt`, audited |

The remaining learning-rate, loss, and augmentation arguments are identical
to the successful final-model recipe in `tools/train_ca_scam.py`. Training is
a direct `YOLO.train(...)` call in the active Colab kernel; it is never placed
in a subprocess.

## Evaluation and decision

Use the same validation split and record Precision, Recall, mAP50, AP75, and
mAP50-95 from `best.pt`. Do not augment validation/test data, move samples
between splits, or inspect the sealed test set during model selection.

Primary comparison: CA-SCAM versus the successful SCAM model at identical
resolution, schedule, initialization route, and data split. A practically
useful result should improve mAP50-95 without materially reducing Recall.
Because this is one training seed, small changes should be described as
preliminary rather than conclusive.

## Reproduction

```powershell
python tools/check_calibrated_scam_models.py --weights yolo11n.pt
pytest -q
git diff --check
```

The Colab notebook is
`colab/YOLO11n_CA_SCAM_VGUP_Training.ipynb`.

## Attribution

The retained SCAM flow is adapted from the official FFCA-YOLO repository:
<https://github.com/yemu1138178251/FFCA-YOLO>. The local-contrast bounded
calibration is the experiment-specific change in this repository.
