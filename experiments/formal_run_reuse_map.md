# Formal run reuse map

One canonical run represents one unique combination of model topology,
dataset, training protocol, and seed. Paper aliases never launch a second
training job.

| Canonical run | Paper aliases | Reused in |
|---|---|---|
| R00 | A0, D0 | cumulative ablation, DPLS baseline, YOLO11 cross-model baseline |
| R01 | D1 | PLS nearest-only comparison |
| R02 | A1, D2, C0 | cumulative ablation, complete DPLS, CA-SCAM no-attention control |
| R03 | C1, CI0 | original SCAM comparison and CA-SCAM internal baseline |
| R04 | A2, C2, CI3, V0 | cumulative ablation, complete CA-SCAM, ERUP/VGUP no-preprocessor control |
| R05A | CI1 | fixed-beta local-contrast calibration |
| R05B | CI2 | learnable unbounded-beta calibration |
| R06 | V1 | original ERUP comparison |
| R07 | VG0 | VGUP with neither gate |
| R08 | VG1 | VGUP global gate only |
| R09 | VG2 | VGUP spatial gate only |
| R10 | A3, V2, VG3 | final YOLO11n, complete VGUP, cumulative final |
| R11 | M0 | official YOLOv8n cross-model baseline |
| R12 | M1 | InceptionDW + DPLS + CA-SCAM + VGUP adapted to YOLOv8n-native C2f |
| R13 | M2 | official YOLO11s model-capacity baseline |
| S00 | S0 | second-dataset independent baseline |
| S01 | S1 | second-dataset independent final method |

The main-data seed-0 matrix therefore contains 15 unique model runs
(R00–R13, counting R05A and R05B separately). S00 and S01 add two runs only
after the second dataset is selected. Stability analysis adds seed 1 and
seed 2 to R00, R02, and R10; these are independent run instances under the
same canonical topology, not new paper aliases.

Cross-dataset zero-shot evaluations are evaluations of existing checkpoints,
not training runs. They must be labeled separately and must never replace
S00/S01 independent training.
