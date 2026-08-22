# PLS multi-scale fusion experiments

All three configurations preserve the D1 PLS backbone and P2/P3/P4 detection
levels while exchanging only the neck fusion graph. They do not import or
modify DySample or any DPLS implementation.

- `yolo11n_pls_asff.yaml` applies one ASFF fusion module to each P2/P3/P4
  output. It follows the feature-resize, three spatial weight maps, softmax
  normalization, weighted summation, and output-refinement path in the ASFF
  paper and the author's `models/network_blocks.py` implementation.
- `yolo11n_pls_scgbifpn.yaml` maps SCGBiFPN's bidirectional flow and direct
  shallow spatial-context skips to P2/P3/P4. Raw P2 is routed directly into
  the final P3 and P4 fusion nodes; no BiFPN scalar weighting is introduced.
- `yolo11n_pls_mafpn.yaml` maps the official MAFPN SAF and AAF graphs to
  P2/P3/P4. SAF reduces the shallow assisted branch, whereas AAF uses
  equal-width dense auxiliary branches. RepHELAN, GHSK, backbone, head, and
  loss changes are intentionally excluded.

Primary sources:

- ASFF: `ruinmessi/ASFF`, commit `4df6f7288b7882a45b8c2dcc3e6e7b499d6cc883`.
- SCGBiFPN: ESL-YOLO, Remote Sensing 2024, 16(23), 4374, Section 3.2 and Figure 5.
- MAFPN: `yang-0201/MAF-YOLO`, commit `e24674cd9ef85b75606e88d7b4d9df7fe4022e1c`.
