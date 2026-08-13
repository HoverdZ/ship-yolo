# ship-yolo

Official experiment repository for a parameter-efficient remote-sensing ship detector based on Ultralytics YOLO11n. The paper-facing model combines:

- **VGUP**: visibility-gated, detection-oriented input processing;
- **InceptionDW**: shallow P2/P3 spatial modeling inside C3k2 bottlenecks;
- **DPLS**: a P2-P4 detection pyramid with DySample upsampling;
- **CA-SCAM**: contrast-aware spatial context calibration before Detect.

The complete YOLO11n model configuration is [`experiments/formal_ablation_v1/A5_inceptiondw_dpls_ca_scam_vgup.yaml`](experiments/formal_ablation_v1/A5_inceptiondw_dpls_ca_scam_vgup.yaml). The YOLOv8n transfer configuration is [`experiments/formal_models/R12_yolov8n_inceptiondw_dpls_ca_scam_vgup.yaml`](experiments/formal_models/R12_yolov8n_inceptiondw_dpls_ca_scam_vgup.yaml).

## Paper release scope

`main` contains the implementations and experiment material reported in the manuscript:

- cumulative YOLO11n ablations and formal training notebooks;
- shallow convolution comparison (standard convolution, InceptionDW, Pinwheel PConv and LSKConv);
- PLS/DPLS, SCAM/CA-SCAM and VGUP mechanism studies;
- YOLOv8n transfer and YOLO11s capacity comparison;
- LS-SSDD-v1.0 and HRSC2016-MS transfer workflows;
- D-FINE-N, SHIP-YOLO, PMF-YOLOv8, E-WFF Net and AC-YOLO comparisons;
- paper-analysis and artifact-generation utilities.

Unselected early experiments are preserved, with their original commit tips, on [`archive/experimental-exploration`](../../tree/archive/experimental-exploration). They are intentionally absent from the current `main` tree.

## Repository layout

```text
custom_modules/                 paper model and comparison modules
experiments/                    model YAMLs and controlled protocols
notebooks/formal/               formal Colab workflows
tools/formal_experiments/       training and dataset preparation helpers
tools/paper_artifacts/          evaluation, analysis and figure/table tools
analysis/ship_scale/            deterministic ship-scale analysis artifacts
paper_artifacts/model_weights/  released checkpoints (Git LFS)
paper_artifacts/training_logs/  released results.csv-style logs
datasets/Fog-LEVIR-Ship/        dataset provenance and reconstruction notice
```

## Environment and training controls

The formal Ultralytics experiments use:

- Ultralytics `8.4.92`;
- input size `640`;
- batch size `8`;
- 150 epochs for the primary formal comparisons;
- seed `0` for the reported run;
- the official matching YOLO pretrained checkpoint for initialization;
- foreground `YOLO.train(...)` execution in the notebook kernel;
- validation-only model selection and `augment=False` for validation/test.

The full shared protocol is recorded in [`experiments/formal_training_config.yaml`](experiments/formal_training_config.yaml). Notebooks copy Drive data to Colab-local storage with concurrent `shutil.copyfile` workers and live file/byte progress before training.

## Checkpoints and logs

The files supplied for the paper release are retained under their original filenames:

- [`paper_artifacts/model_weights`](paper_artifacts/model_weights)
- [`paper_artifacts/training_logs`](paper_artifacts/training_logs)

Clone checkpoints with Git LFS enabled:

```bash
git lfs install
git clone https://github.com/HoverdZ/ship-yolo.git
```

## Dataset notice

The primary dataset is referred to as **Fog-LEVIR-Ship**. It is derived from the public **LEVIR-Ship** dataset (the name is sometimes mistyped as “Liver_Ship”) and applies the cloud/fog simulation procedure of Wang et al. (2022), including Perlin-noise-based thin, dense and patchy fog; exact construction details are described in the paper.

The image files are not redistributed in this public repository. The LEVIR-Ship authors license their annotations under CC BY 4.0 but state that they do not own the source satellite-image copyright. Users must obtain the source data through the official channel and comply with the applicable satellite-data terms. See [`datasets/Fog-LEVIR-Ship/README.md`](datasets/Fog-LEVIR-Ship/README.md).

## Module registration

The project does not vendor or modify the installed Ultralytics package. Register repository modules before constructing a custom YAML:

```python
from custom_modules.register import register_custom_modules
from ultralytics import YOLO

register_custom_modules()
model = YOLO("experiments/formal_ablation_v1/A5_inceptiondw_dpls_ca_scam_vgup.yaml")
```

## Research-use note

Several adapted comparison modules retain or depend on their upstream license conditions. Review upstream licenses before commercial reuse. No dataset license or third-party model license is replaced by this repository.
