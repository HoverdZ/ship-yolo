# SA-DWPN-C-lite 640 Main Result

Source: user-reported validation output  
Recorded at: 2026-07-08

## Structural Difference

SA-DWPN-C-lite is based on SA-DWPN-B and enables spatial gate only at the T3 and O3 SDWF nodes. T4, O4, and O5 keep spatial gate disabled. Detect remains Detect(O3, O4, O5), corresponding to P3/P4/P5.

## Unified Protocol

Formal SA-DWPN ablations should follow `configs/sa_dwpn_protocol.yaml`: imgsz 640, 150 epochs, batch 8, seed 0, deterministic mode enabled, official `yolo11n.pt` initialization, same dataset split, and one structural variable changed per ablation.

## Current Results

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| SA-DWPN-B | 0.83527 | 0.72529 | 0.77704 | 0.31029 |
| SA-DWPN-C-lite | 0.839 | 0.650 | 0.745 | 0.296 |

Model summary for C-lite:

- fused layers: 105
- parameters: 3,351,842
- GFLOPs: 8.7
- pretrained transfer: 242/481
- validation images: 842
- validation instances: 688

## Metric Change

- Precision: +0.00373
- Recall: -0.07529
- mAP50: -0.03204
- mAP50-95: -0.01429

## Current Conclusion

C-lite slightly improves Precision but reduces Recall, mAP50, and mAP50-95 compared with B. This suggests C-lite is not yet a better main model under the current reported validation result.

## Unverified Hypothesis

The current research hypothesis is that enabling spatial gate at both T3 and O3 may over-suppress weak small-ship responses. This is only a hypothesis and must not be treated as a proven causal fact.

## Next Steps

- Run T3-only ablation.
- Run O3-only ablation.
- Inspect eta values and spatial masks.
- Generate heatmap comparisons as auxiliary interpretation only.
- Export real `args.yaml`, `results.csv`, and figures from Google Drive using `tools/export_run_artifacts.py`.
