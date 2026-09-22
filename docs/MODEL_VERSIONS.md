# Model identity and version guide

## Current model: SIP-Det

**SIP-Det** expands to **Ship Information Preservation Detector** (船舶信息保留检测器). The name describes the design objective of preserving useful ship cues through input processing, scale sampling and context/detail refinement. It does not assert lossless information transmission or an information-theoretic guarantee.

Canonical architecture: [M7](../experiments/final_ablation/M7_yolo11n_vgup_ldpp_cgdr.yaml), built on YOLO11n with VGUP, LDPP and CGDR.

## Evolution and compatibility

| Family | Status | Meaning |
|---|---|---|
| InceptionDW + DPLS + CA-SCAM + VGUP | Historical | Earlier architecture formerly linked as final from the root README |
| VGUP + DPLS + CGDR | Historical precursor | Earlier scale-path design with CGDR; not the complete LDPP model |
| VGUP + LDPP + CGDR | Current canonical model | SIP-Det, identified by M7 |
| Host-specific LDPP transfers | Adaptations | Follow the separate architecture transfer guide; graphs need not be identical to M7 |

LDPP includes the P2/P3/P4 scale path with dynamic upsampling, selected depthwise neck operations, and the finalized scale-specific LDPPDetect regression towers. A DPLS configuration or checkpoint does not become LDPP by changing its name.

The canonical model contains no InceptionDW or CA-SCAM node. Those classes remain available for historical experiments. The filename `custom_modules/dpls_lightweight_convs.py` is retained because current registration and historical configurations rely on its import path.

## Experiment and artifact identity

- Preserve original filenames, configuration graphs and logs for historical runs.
- Use the [factorial ablation map](../experiments/final_ablation/README.md) to identify current factor combinations.
- Use SIP-Det only for runs whose architecture matches canonical M7; record dataset, class count, initialization, training protocol and configuration revision with metrics.
- Do not relabel historical DPLS results as LDPP or SIP-Det results.
- A checkpoint filename such as `最终模型.pt` or `YOLOv8n_ours.pt` alone does not establish correspondence to M7. Verify its embedded architecture and run provenance before publishing it as a SIP-Det checkpoint.
- YOLOv13 transfer retains P5 and uses KBL-Lite rather than full VGUP. See the [host-specific guide](../experiments/architecture_transferability/README.md).

This documentation update changes model identity and navigation only. Existing architecture definitions, source import paths, checkpoints and numerical results are preserved.
