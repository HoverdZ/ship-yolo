# D-FINE-N complexity–accuracy baseline

## Purpose

This run adds the official D-FINE Nano detector as an external
complexity–accuracy baseline. It is not a structural adaptation of YOLO and
does not reuse any ship-experiment checkpoint.

## Official implementation

- Repository: https://github.com/Peterande/D-FINE
- Pinned commit: `7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6`
- Official model: D-FINE-N with HGNetv2-B0.
- Official custom-data base config:
  `configs/dfine/custom/dfine_hgnetv2_n_custom.yml`.
- Initialization: official COCO D-FINE-N checkpoint
  `dfine_n_coco.pth`.
- The implementation is cloned at runtime; third-party source is not copied
  into this repository.

The official model card reports approximately 4 M parameters and 7 GFLOPs,
but the paper table must use the locally generated complexity artifact for
the exact runtime config. Official TensorRT latency is not mixed with local
PyTorch latency.

## Controlled comparison

The frozen primary train/validation/test split is unchanged. The source
YOLO-HBB labels are deterministically converted to COCO without moving images
between splits. Because the official D-FINE custom loader uses raw category
ids when `remap_mscoco_category=False`, the single ship category is encoded
as id 0. Validation selects the checkpoint; the test split remains sealed.

The cross-model controls are:

- input: 640 × 640;
- epochs: 150;
- total batch size: 8 on one GPU;
- seed: 0;
- AMP enabled;
- official COCO pretraining;
- the same frozen dataset split.

D-FINE retains its official AdamW, EMA, two-stage augmentation, loss, and
decoder recipe. The Nano custom config specifies main/backbone learning rates
of 8e-4/4e-4 at total batch 128. Following the official README's linear
scaling rule, total batch 8 uses 5e-5/2.5e-5. Warmup durations are scaled by
the inverse batch ratio so that warmup exposure is preserved in samples.
Stage 1 ends at epoch 140, leaving the final ten epochs for the official
stage-2 refinement.

## Execution and artifacts

Notebook:
`notebooks/formal/DFINE_N_Complexity_Tradeoff.ipynb`.

Training is a direct foreground call to the official `train.main()` in the
current Colab kernel. Setup commands may use subprocesses, but model training
does not. A background file-mirroring thread copies only stable checkpoints
to Drive and never runs training.

The run directory is:
`/content/drive/MyDrive/ship_detection/paper_project/formal_experiments/DFINE_N/seed_0`.

Required final artifacts include the official-repository commit, runtime
config, dataset conversion report, official checkpoint SHA-256, explicit
Loaded/Total tensor audit, log, best checkpoint, P/R, AP50, AP75, AP50–95,
parameter count, local complexity/latency metadata, environment record, and
checksum manifest.
