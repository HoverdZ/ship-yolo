# Released experiment artifacts

The files in this directory were supplied for the paper release and retain their original filenames.

- `model_weights/`: YOLO checkpoints (`.pt`) and the D-FINE-N checkpoint (`.pth`), stored through Git LFS.
- `training_logs/`: Ultralytics-style CSV logs and the D-FINE-N text log.

These artifacts correspond to the formal models, ablations, cross-architecture/cross-dataset runs and comparison methods discussed in the manuscript. Use the matching code path in `experiments/`, `notebooks/formal/` and `custom_modules/` when loading a custom checkpoint.
