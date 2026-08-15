# Small and Tiny Ship Detection in Wide-Area Ocean Remote Sensing

This repository implements a parameter-efficient visibility-scale-context collaborative detection method for wide-area ocean remote sensing imagery. Built on Ultralytics YOLO11n, the method improves small and tiny ship detection under complex marine conditions through shallow spatial feature extraction, small-object-oriented detection-scale reconstruction, context calibration, and visibility-aware input processing.

## Core Components

- **InceptionDW**: Adapted to the shallow C3k2 bottlenecks for multi-scale spatial feature extraction at the P2 and P3 backbone stages.
- **DPLS**: Reconstructs a P2-P4 detection pyramid to preserve fine-grained features for small and tiny objects.
- **CA-SCAM**: Performs contrast-aware spatial context calibration for ships under low-visibility conditions.
- **VGUP**: Applies adaptive input enhancement according to image visibility.

## Repository Structure

- `custom_modules/`: Implementations and registration code for the proposed modules.
- `experiments/`: Model YAML files and experiment configurations.
- `model_weights/`: Trained model weights managed with Git LFS.
- `training_logs/`: Training logs and metric records.
- `datasets/`: Dataset provenance and usage documentation.

## Final Model Configuration

The complete model configuration is available at:

```text
experiments/model_ablation/A5_inceptiondw_dpls_ca_scam_vgup.yaml
```

Main training settings:

- Ultralytics: `8.4.92`
- Input size: `640 x 640`
- Batch size: `8`
- Training epochs: `150`
- Random seed: `0`

Register the custom modules before constructing the model:

```python
from custom_modules.register import register_custom_modules
from ultralytics import YOLO

register_custom_modules()
model = YOLO("experiments/model_ablation/A5_inceptiondw_dpls_ca_scam_vgup.yaml")
```

## RTMDet-Tiny Cross-Architecture Evaluation

The paired RTMDet-Tiny experiment keeps the native MMDetection detector,
assignment strategy, losses, head, optimizer family, and augmentation recipe.
Only the proposed visibility-scale-context components change between the two
configs:

- `experiments/transfer_models/X05_rtmdet_tiny_official_150e.py`
- `experiments/transfer_models/X06_rtmdet_tiny_inceptiondw_dpls_ca_scam_vgup_150e.py`

Both use 640 × 640 input, 150 epochs, batch size 8, seed 0, and the same
official RTMDet-Tiny COCO checkpoint. The runtime records exact
Loaded/Total tensor counts before epoch one.

## Dataset

The primary experiments use a fog-augmented version of the public LEVIR-Ship dataset. Following Wang et al. (2022), Perlin noise is used to simulate thin, dense, and patchy fog conditions. Dataset provenance, processing details, and directory organization are documented in [`datasets/Fog-LEVIR-Ship/README.md`](datasets/Fog-LEVIR-Ship/README.md).

The original remote-sensing images are not redistributed in this repository. Users should obtain the source dataset in accordance with its original terms and reproduce the derived data using the documented procedure.

## References

- Chen, W. et al. LEVIR-Ship: [https://github.com/WindVChen/LEVIR-Ship](https://github.com/WindVChen/LEVIR-Ship)
- Wang, W., Zhang, X., Sun, W., and Huang, M. (2022). *A Novel Method of Ship Detection under Cloud Interference for Optical Remote Sensing Images*. Remote Sensing, 14(15), 3731. [https://doi.org/10.3390/rs14153731](https://doi.org/10.3390/rs14153731)
