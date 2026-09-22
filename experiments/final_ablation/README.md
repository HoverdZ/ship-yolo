# SIP-Det: complete component ablation index

The eight existing configurations below form the VGUP/LDPP/CGDR factorial design. Files are kept at their original paths to preserve notebook and run references. A/M identifiers are existing file identifiers, not new experimental results.

| ID | VGUP | LDPP | CGDR | Configuration |
|---|---|---|---|---|
| A0 | — | — | — | [A0_yolo11n.yaml](../ldpp_ablation/A0_yolo11n.yaml) |
| A1 | ✓ | — | — | [A1_yolo11n_vgup.yaml](../ldpp_ablation/A1_yolo11n_vgup.yaml) |
| M2 | — | ✓ | — | [M2_yolo11n_ldpp.yaml](M2_yolo11n_ldpp.yaml) |
| A3 | — | — | ✓ | [A3_yolo11n_cgdr.yaml](../ldpp_ablation/A3_yolo11n_cgdr.yaml) |
| A4 | ✓ | — | ✓ | [A4_yolo11n_vgup_cgdr.yaml](../ldpp_ablation/A4_yolo11n_vgup_cgdr.yaml) |
| M5 | — | ✓ | ✓ | [M5_yolo11n_ldpp_cgdr.yaml](M5_yolo11n_ldpp_cgdr.yaml) |
| M6 | ✓ | ✓ | — | [M6_yolo11n_vgup_ldpp.yaml](M6_yolo11n_vgup_ldpp.yaml) |
| M7 | ✓ | ✓ | ✓ | [M7_yolo11n_vgup_ldpp_cgdr.yaml](M7_yolo11n_vgup_ldpp_cgdr.yaml) |

**M7 is the canonical SIP-Det architecture.**

With LDPP enabled, prediction uses P2/P3/P4 at strides 4/8/16 and LDPPDetect. With LDPP disabled, prediction retains P3/P4/P5 at strides 8/16/32 and the native Detect head. CGDR replaces SPPF at the deepest retained stage: P4 with LDPP, P5 without it.

LDPP includes the P2–P4 scale path, two DySample nodes, two selected C3k2_DWConvLite fusion blocks, two direct DWConv transitions and scale-specific regression allocation. The [allocation study](../ldpp_allocation_study/README.md) separately examines that regression design.

Use the [shared protocol](../training_config.yaml) for experiment settings. This index establishes configuration identity, not completion of training or verification of published metrics. Old DPLS experiments are indexed separately in the [experiment guide](../README.md).
