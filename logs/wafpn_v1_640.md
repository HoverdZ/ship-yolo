# Experiment Record: WAFPN-v1-640

## Basic Info

Experiment Name: wafpn_v1_640
Model YAML: experiments/yolo11n_wafpn_v1_640.yaml
Resolution: 640
Detect Heads: 3 (P3/P4/P5)

## Motivation

WAFPN-v1 is tested after Baseline/P2/SemanticGuide experiments because current observations suggest that high-resolution information helps remote sensing ship detection, while direct shallow detection can introduce noise. This experiment keeps the three-head YOLO11 detection design and evaluates whether learnable weighted feature fusion is a better Neck-level alternative.

## Modification

Replace the 4 Neck fusion nodes with `AlignWeightedAdd2`:

1. P5 upsample + P4 -> P4_fused
2. P4_fused upsample + P3 -> P3_fused
3. P3_fused downsample + P4_fused -> P4_out
4. P4_out downsample + P5 -> P5_out

## Weight Transfer

Pretrained: TBD
Loaded/Total tensors: TBD
Mismatch layers: TBD
Random initialized modules: WeightedAdd / AlignConv / TBD

## Training Setup

Epoch: TBD
imgsz: 640
batch: TBD
device: TBD

## Results

Precision: TBD
Recall: TBD
mAP50: TBD
mAP50-95: TBD

## Analysis

TBD

## Conclusion

TBD

## Next Step

TBD
