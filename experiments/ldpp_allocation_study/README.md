# LDPP regression allocation study

This study isolates only the scale-aware convolution allocation inside the
LDPP regression tower. The YOLO11n backbone, DPLS P2/P3/P4 pyramid, final LDPP
lightweight neck, classification branch and all detection behavior outside
`cv2` are fixed.

| ID | P2 | P3 | P4 | Experiment configuration |
|---|---|---|---|---|
| H0 | DS+DS | DS+DS | DS+DS | `H0_yolo11n_ldpp_all_ds.yaml` |
| H1 | Dense+DS | DS+DS | DS+DS | `H1_yolo11n_ldpp_p2_dense.yaml` |
| H2 | Dense+DS | Dense+DS | DS+DS | `../final_ablation/M2_yolo11n_ldpp.yaml` |
| H3 | Dense+DS | Dense+DS | Dense+DS | `H3_yolo11n_ldpp_all_scale_dense_first.yaml` |
| H4 | Dense+Dense | Dense+Dense | Dense+Dense | `H4_yolo11n_ldpp_all_dense.yaml` |
| H5 | DS+Dense | DS+Dense | DS+DS | `H5_yolo11n_ldpp_p23_dense_second.yaml` |

H2 is the canonical `LDPPDetect` and reuses the existing M2 YOLO11n LDPP
experiment. It is neither duplicated nor modified here. H0, H1, H3, H4 and H5
are the five new controlled experiments.

## Frozen definition

- Dense: `Ultralytics Conv(in, out, k=3, s=1)`.
- DS: the exact canonical LDPP block, `DWConv(in, in, k=3, s=1)` followed by
  `Conv(in, out, k=1, s=1)`.
- Predictor: `nn.Conv2d(reg_hidden_channels, 4 * reg_max, kernel_size=1)`.
- Hidden width: `max(16, P2_channels // 4, reg_max * 4)`.
- `reg_max=16`; inputs are P2/P3/P4 at strides 4/8/16.

All variants inherit Ultralytics `Detect`. Parent-created stock `cv3`
classification is retained without reconstruction. DFL, forward, decoding,
loss and end-to-end behavior are inherited; when end-to-end is enabled, the
final variant `cv2` is deep-copied to `one2one_cv2` just as in canonical LDPP.

Every new YAML is a byte-for-byte copy of
`experiments/final_ablation/M2_yolo11n_ldpp.yaml` except for the final Detect
module name. Therefore VGUP and CGDR remain off; the two DySample blocks, two
`C3k2_DWConvLite` blocks, two direct DWConv transitions and Detect sources
`[14, 17, 20]` are identical. No differences in backbone, neck, channels,
classification, predictor, loss or target assignment belong in this study.

## Minimal validation

The intended check is limited to Python compilation, topology equality, and
one construction of each H0/H1/H3/H4/H5 model after
`register_custom_modules()`. No extra random forward, backward pass, training,
benchmark or FLOP profiling is part of this structural study.
