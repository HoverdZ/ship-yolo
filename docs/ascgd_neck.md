# YOLO11n-InceptionDW ASCGD-Neck experiments

## Scope

This experiment suite keeps the validated InceptionDW backbone implementation
and its layer-2/layer-4 placement unchanged. Only the neck changes. Every
variant retains the official three-scale YOLO11 Detect head at strides 8, 16,
and 32; no P2 head, custom loss, external CUDA operator, FaPN module, or
deformable convolution is used.

ASCGD means **Asymmetric Spatial–Channel Gather-and-Distribute**. Copies of C3,
C4, and C5 are aligned to 128 channels at P4 resolution. One dual aggregation
block applies 8x8 window spatial self-attention, cross-covariance channel
self-attention, adaptive interaction, and SGFN. The result remains anchored by
the aligned C4 feature:

```text
A3 = PWConv(SiLU(BN(DWConv3x3-s2(C3)))))
A4 = SiLU(BN(Conv1x1(C4)))
A5 = Bilinear2x(SiLU(BN(Conv1x1(C5))))
G0 = SiLU(BN(Conv1x1(Concat(A3, A4, A5))))
G  = A4 + DualAggregation(G0)
```

The full E variant sends public semantics to P3 with window spatial
cross-attention and sends shallow geometry to P4/P5 with channel
cross-attention. All injections are gated residuals initialized to 0.1 and
followed by ordinary YOLO C3k2 refinement.

## Seven controlled variants

| Variant | Gather center | P3 | P4 | P5 |
|---|---|---|---|---|
| A | No | Original YOLO FPN/PAN | Original YOLO FPN/PAN | Original YOLO FPN/PAN |
| B | Yes | Direct | Direct | Direct |
| C | Yes | Spatial cross | Direct | Direct |
| D | Yes | Direct | Channel cross | Channel cross |
| E | Yes | Spatial cross | Channel cross | Channel cross |
| F | Yes | Channel cross | Spatial cross | Spatial cross |
| G | Yes | Spatial + channel | Spatial + channel | Spatial + channel |

A is structurally identical to
`experiments/yolo11n_inceptiondw_c3k2_p23.yaml`. The automated audit compares
layer structure, parameter count, GFLOPs, and every state-dictionary key and
shape. All seven YAMLs share the same first 11 backbone layers.

The historical baseline YAML stores the generic 80-class placeholder, but the
formal ship dataset overrides it to one class. A-G therefore declare `nc=1`,
and A is compared against the existing InceptionDW YAML rebuilt with the same
formal one-class override. This makes the preflight model, weight-transfer
report, and model actually handed to Trainer identical instead of deferring a
Detect rebuild until training starts.

## Initialization

Formal training starts from official `yolo11n.pt`. Backbone tensors are
inherited only when name and shape match. The shifted Detect layer is mapped
explicitly by semantic role and exact shape. InceptionDW-only and ASCGD neck
parameters retain their normal initialization. No tensor is cropped, repeated,
or padded.

`--init-from-inception-best` exists only for debugging and must be omitted from
formal comparisons.

```bash
python tools/init_ascgd_weights.py --variant e_full --weights yolo11n.pt
```

## CPU preflight

```bash
python tools/build_ascgd_variants.py --variant all
python tools/check_ascgd.py --all
python -m pytest tests/test_ascgd.py -q
```

The complete check writes:

```text
reports/ascgd_preflight/summary.json
reports/ascgd_preflight/model_comparison.csv
reports/ascgd_preflight/weight_transfer.json
reports/ascgd_preflight/module_inventory.json
reports/ascgd_preflight/check_report.md
```

Ultralytics/THOP GFLOPs can omit explicit attention matrix multiplication
costs. The report records that limitation, and any profiling exception is
stored verbatim rather than ignored.

## First formal Colab run

Install the pinned runtime and run E first:

```bash
pip install ultralytics==8.4.92
python tools/train_ascgd_colab.py \
  --variant e_full \
  --data /content/datasets/ship/data.yaml \
  --project /content/drive/MyDrive/ship_detection/organized_experiments \
  --name yolo11n_incdw_ascgd_full_640
```

The entrypoint requires a one-class dataset, prints differences from the
expected 2582/842/874 split counts, refuses to overwrite existing directories,
verifies every Trainer parameter against the audited initialization before the
first optimizer step, and validates `best.pt` after training. Resume only from the exact saved
optimizer/scheduler checkpoint:

```bash
python tools/train_ascgd_colab.py \
  --variant e_full \
  --data /content/datasets/ship/data.yaml \
  --project /content/drive/MyDrive/ship_detection/organized_experiments \
  --name yolo11n_incdw_ascgd_full_640 \
  --resume
```

The shared formal policy is stored in `configs/ascgd_experiments.yaml`.
The repository has no completed InceptionDW `args.yaml`; therefore documented
InceptionDW settings are retained and the remaining values are pinned to the
Ultralytics 8.4.92 defaults already used by the repository. This provenance is
recorded explicitly instead of presenting inferred values as historical facts.

## L4 benchmark and result summary

```bash
python tools/benchmark_ascgd.py --variant all --warmup 50 --iterations 200
python tools/summarize_ascgd_results.py \
  --runs /content/drive/MyDrive/ship_detection/organized_experiments/yolo11n_incdw_ascgd_full_640 \
  --benchmark reports/ascgd_preflight/l4_benchmark.json
```

The benchmark uses batch 1, 640 images, unfused PyTorch eager inference,
explicit CUDA synchronization, and reports FP32/AMP mean, P50/P95,
throughput, and peak memory. No local formal training or GPU benchmark is
performed during code preparation.
