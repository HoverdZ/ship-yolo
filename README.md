# SIP-Det: Ship Information Preservation Detector

SIP-Det is a YOLO11n-based detector for small and tiny ships in complex ocean remote sensing imagery. Its design addresses degradation of useful ship information along the detection pipeline: limited input visibility, spatial dilution during scale sampling, and imbalance between contextual information and local detail.

## Current final model

**SIP-Det = YOLO11n + VGUP + LDPP + CGDR.**

The authoritative configuration is [M7_yolo11n_vgup_ldpp_cgdr.yaml](experiments/final_ablation/M7_yolo11n_vgup_ldpp_cgdr.yaml). The model name identifies this existing architecture; it does not introduce another model variant.

| Component | Role | Implementation |
|---|---|---|
| VGUP | Visibility-Gated Unified Processing: shared lightweight parameter encoder, global BPW residual gate and spatial KBL residual gate | [vgup.py](custom_modules/vgup.py) |
| LDPP | P2/P3/P4 detection pyramid with DySample, selected lightweight neck operations and scale-specific regression towers | [M7 configuration](experiments/final_ablation/M7_yolo11n_vgup_ldpp_cgdr.yaml), [neck operators](custom_modules/dpls_lightweight_convs.py), [LDPPDetect](custom_modules/ldpp_detect.py) |
| CGDR | Context-Guided Detail Refinement: context aggregation and gated local detail supplementation | [cgdr.py](custom_modules/cgdr.py) |

VGUP processes the RGB input before the backbone. LDPP removes the P5 backbone stage and organizes prediction at strides 4, 8 and 16. CGDR operates at the deepest retained P4 backbone stage, before C2PSA and neck fusion. LDPP spans the scale hierarchy, neck and heads; the three components are not three consecutive standalone blocks.

LDPP uses Dense+DS regression blocks at P2/P3 and DS+DS at P4. Dense denotes standard 3×3 convolution; DS denotes depthwise 3×3 followed by pointwise 1×1 convolution. The native classification branch is retained.

## Construct the final model

Run from the repository root in the intended Ultralytics 8.4.92 environment:

```python
from custom_modules.register import register_custom_modules
from ultralytics import YOLO

register_custom_modules()
model = YOLO(
    "experiments/final_ablation/M7_yolo11n_vgup_ldpp_cgdr.yaml",
    task="detect",
)
```

This constructs the architecture, not a trained detector. The YAML retains the existing `nc: 80` template value; ship training must use a dataset YAML declaring the actual ship classes. Ultralytics training adapts the class count to the dataset.

The recorded shared protocol uses 640×640 inputs, batch size 8, 150 epochs and seed 0. See [training_config.yaml](experiments/training_config.yaml) for the full protocol, initialization policy and environment requirements.

## Experiments and versions

- [Experiment index](experiments/README.md): current ablations, component studies, transfer configurations and historical experiments.
- [Complete three-factor ablation map](experiments/final_ablation/README.md): all eight VGUP/LDPP/CGDR combinations, referencing the existing files.
- [Version and naming guide](docs/MODEL_VERSIONS.md): DPLS versus LDPP and the scope of SIP-Det.
- [Architecture transfer guide](experiments/architecture_transferability/README.md): host-specific adaptations and their differences.

InceptionDW + DPLS + CA-SCAM + VGUP is an earlier architecture. Its configurations, logs and weights remain historical records and are not the current final model. DPLS is a precursor to LDPP, not an interchangeable name. The source filename `dpls_lightweight_convs.py` remains valid: its operators are reused by LDPP.

## Repository layout

| Directory | Contents |
|---|---|
| `custom_modules/` | Current modules, shared operators and historical/comparison implementations |
| `experiments/` | Architecture configurations and protocols, classified in the experiment index |
| `docs/` | Model identity and version documentation |
| `model_weights/` | Git LFS checkpoints; architecture provenance must be verified before assigning a model label |
| `training_logs/` | Original run records; historical metrics retain their original model identity |
| `datasets/` | Dataset provenance and preparation documentation |

## Dataset and attribution

Primary experiments use a fog-augmented version of LEVIR-Ship. See [dataset documentation](datasets/Fog-LEVIR-Ship/README.md) for provenance and processing details. Original remote sensing imagery is not redistributed.

VGUP builds on the BPW/KBL filters of [ERUP-YOLO](https://arxiv.org/abs/2411.02799). LDPP uses [DySample](https://github.com/tiny-smart/dysample). CGDR reuses the HHSPP context operator migrated from DPCSANet, as documented in [hhspp.py](custom_modules/hhspp.py). These reused operators should be distinguished from the proposed adaptations.

- LEVIR-Ship: [original dataset repository](https://github.com/WindVChen/LEVIR-Ship).
- Wang et al. (2022), *A Novel Method of Ship Detection under Cloud Interference for Optical Remote Sensing Images*, Remote Sensing 14(15), 3731. [DOI](https://doi.org/10.3390/rs14153731).
