# YOLO11n C3Cross-P23 short screening

## Motivation

The completed full-backbone C3Cross run reached its best validation result at
epoch 148:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.81172 | 0.69767 | 0.76795 | 0.32184 |

The comparable YOLO11n baseline best result was:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.81490 | 0.71667 | 0.77364 | 0.32074 |

Full C3Cross therefore gained only 0.00110 mAP50-95 while losing 0.01900
Recall and 0.00569 mAP50. This experiment tests whether the shallow
P2/P3 blocks retain useful localization behavior without changing the deeper
P4/P5 semantic blocks.

## Exact structural change

- Preserve backbone layers 1 and 3 as ordinary `3x3, stride=2` Ultralytics
  `Conv` layers.
- Replace only the following C3k2 blocks at layers 2 and 4 with
  `C3k2CrossConv`.
- Keep P4/P5 C3k2, SPPF, C2PSA, Neck, and the P3/P4/P5 Detect head unchanged.

The protected stride-2 convolutions are not inside C3k2 in YOLO11n. They are
independent layers immediately before the P2 and P3 C3k2 stages. Automated
tests inspect their kernel and stride directly.

## Initialization

This is a low-cost screening run, not a same-budget formal ablation:

1. Build `experiments/yolo11n-c3cross-p23.yaml` using the dataset class count.
2. Load every exact-name, exact-shape tensor from the trained YOLO11n baseline.
3. Overwrite every `model.2.*` and `model.4.*` tensor from the trained
   full-backbone C3Cross checkpoint.
4. Require complete target coverage and record the final tensor provenance.
5. Start training with `pretrained=False` so `yolo11n.pt` cannot overwrite the
   hybrid initialization.

## AP75 audit

Before screening, validate the baseline and full C3Cross `best.pt` checkpoints
with identical settings:

- split: `val`
- imgsz: 640
- batch: 8
- augment: false
- AP75 source: `metrics.box.map75`

The test split remains sealed. Results are written to
`ship_detection/audits/ap75_comparison.csv` on Google Drive.

## Screening protocol

| Setting | Value |
| --- | --- |
| epochs | 30 maximum |
| imgsz | 640 |
| batch | 8 |
| optimizer | AdamW |
| lr0 / lrf | 0.0005 / 0.1 |
| warmup_epochs | 1 |
| mosaic / close_mosaic | 0.5 / 5 |
| scale / translate | 0.4 / 0.1 |
| patience | 10 |
| seed | 0 |
| deterministic | true |

At epoch 15, stop when the best mAP50-95 is below 0.320 or its Recall is below
0.700. Promotion after epoch 30 requires all of:

- mAP50-95 >= 0.324
- Recall >= 0.705
- mAP50 >= 0.770

Failure means C3Cross is abandoned without a new 150-epoch run.

## Optional fine-tune

Only a promoted P2/P3 model may start one independent 20-epoch fine-tune from
its `best.pt`. It uses AdamW, `lr0=0.0003`, `mosaic=0.2`,
`close_mosaic=5`, `scale=0.30`, and `translate=0.05`. Loss weights remain
`box=7.5`, `cls=0.5`, and `dfl=1.5`.

## Reproduction

Use `colab/YOLO11n_C3Cross_P23_Screening.ipynb`. The notebook installs
Ultralytics 8.4.92, securely checks out
`experiment/yolo11n-c3cross-p23`, copies the dataset to `/content` with 16
`shutil.copyfile` workers, runs AP75 and structure audits, and then exposes
separate screening and optional fine-tune cells.
