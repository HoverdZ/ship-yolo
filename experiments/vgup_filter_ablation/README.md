# VGUP filter and gate ablation

Purpose: show whether each filter helps and whether each gate helps.
All rows use the lightweight encoder and the fixed BPW -> KBL order.
A disabled gate is fixed to 1, which is mathematically the plain filter.
A disabled filter is deleted together with its encoder parameters and its gate;
everything else stays unchanged.

| ID | BPW | KBL | Global gate | Spatial gate | Config |
|---|---|---|---|---|---|
| A0 | — | — | — | — | [A0_yolo11n.yaml](../ldpp_ablation/A0_yolo11n.yaml) |
| F1 | ✓ | — | — | — | [F1_yolo11n_bpw.yaml](F1_yolo11n_bpw.yaml) |
| F2 | ✓ | — | ✓ | — | [F2_yolo11n_bpw_gate.yaml](F2_yolo11n_bpw_gate.yaml) |
| F3 | — | ✓ | — | — | [F3_yolo11n_kbl.yaml](F3_yolo11n_kbl.yaml) |
| F4 | — | ✓ | — | ✓ | [F4_yolo11n_kbl_gate.yaml](F4_yolo11n_kbl_gate.yaml) |
| F5 | ✓ | ✓ | — | — | [F5_yolo11n_bpw_kbl.yaml](F5_yolo11n_bpw_kbl.yaml) |
| F5a | ✓ | ✓ | ✓ | — | [F5a_yolo11n_bpw_kbl_global_gate.yaml](F5a_yolo11n_bpw_kbl_global_gate.yaml) |
| F5b | ✓ | ✓ | — | ✓ | [F5b_yolo11n_bpw_kbl_spatial_gate.yaml](F5b_yolo11n_bpw_kbl_spatial_gate.yaml) |
| F6 | ✓ | ✓ | ✓ | ✓ | [F6_yolo11n_bpw_kbl_gates.yaml](F6_yolo11n_bpw_kbl_gates.yaml) |

F6 builds the same network as A1 (`[8, 128, true, true]`), so an existing A1 run
with the shared protocol can be reused. F5 is the same network as the old
gate-study "no gate" row, but built with `VGUPPreprocessor`.

Training settings follow [training_config.yaml](../training_config.yaml).
