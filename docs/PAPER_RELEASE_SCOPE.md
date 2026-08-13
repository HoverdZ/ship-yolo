# Paper release scope

The `main` branch is the paper-facing release. It keeps the current implementations, experiment definitions, notebooks and analysis tools used by the final manuscript.

Historical experiments that were not selected for the manuscript were consolidated on `archive/experimental-exploration`. That branch records each former remote branch and commit tip in `docs/EXPERIMENTAL_BRANCH_ARCHIVE.md`. The archive uses ancestry-preserving merges, so deleted branch names can be recreated from their recorded SHA without placing unused files in the current `main` tree.

## Final method

The final YOLO11n method comprises shallow InceptionDW adaptation, DPLS, bounded CA-SCAM and full VGUP. Its paper-facing configuration is:

```text
experiments/formal_ablation_v1/A5_inceptiondw_dpls_ca_scam_vgup.yaml
```

The paper-reported YOLOv8n transfer is:

```text
experiments/formal_models/R12_yolov8n_inceptiondw_dpls_ca_scam_vgup.yaml
```

## Excluded exploration lines

Examples moved out of the current `main` tree include CrossConv/DD/CGFM, C3Cross, WAFPN, SA-DWPN, the unreported PKIConv screening run and the abandoned APFAN comparison. They remain recoverable from the archive branch.
