# R12: YOLOv8n final-method adaptation

## Experiment identity

- Paper alias: M1.
- Base detector: YOLOv8n.
- Final modules: InceptionDW + DPLS + bounded CA-SCAM + complete VGUP.
- Initialization: independent compatible-tensor transfer from `yolov8n.pt`.
- Previous experiment checkpoints: prohibited.
- Dataset: the same frozen train/validation/test split used by R11 and R13.

## Structural adaptation

The model retains YOLOv8-native C2f blocks and channel scaling rather than
transplanting YOLO11 C3k2 or C2PSA layers.  In the P2 and P3 backbone C2f
blocks, every Bottleneck keeps its first 3x3 convolution and shortcut rule;
only the second 3x3 spatial convolution is replaced by InceptionDW.  The P4
backbone C2f and all neck C2f blocks remain official YOLOv8 components.

DPLS uses two DySample operators and moves detection to P2/P3/P4.  Three
bounded CA-SCAM modules are placed immediately before Detect.  Complete VGUP
is used at the RGB input with both global and spatial gates enabled.

## Frozen training protocol

- epochs: 150
- imgsz: 640
- batch: 8
- optimizer: auto
- seed: 0
- cache: disk
- validation/test augmentation: disabled
- test set: sealed during model selection

The Colab Notebook performs dataset, topology, CPU forward/backward, stride,
complexity, and official-weight-transfer audits before calling the official
Ultralytics `YOLO.train()` API directly in the foreground.
