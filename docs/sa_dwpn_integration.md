# SA-DWPN-B Integration Notes

## Scope

This repository does not vendor the full Ultralytics source code. The files here are prepared as a source-ready research scaffold:

- `custom_modules/sa_dwpn.py`
- `experiments/yolo11n_sa_dwpn_b.yaml`
- `tools/test_sa_dwpn_build.py`
- `tools/test_sa_dwpn_weight_transfer.py`
- `tools/train_sa_dwpn_smoke.py`

To run the model inside a copied Ultralytics source tree, copy and register the module and YAML as described below.

## Module Design

`DWDown` downsamples the C2 detail branch from stride 4 to stride 8 using a stable `Conv(c1, c2, k=3, s=2)`.

`SDWF` implements Scene-Adaptive Dynamic Weighted Fusion:

- static learnable weights initialized to 1
- optional image-level dynamic gate enabled by default
- `gamma` initialized to 0 so the block starts close to static fusion
- optional spatial gate implemented but disabled in SA-DWPN-B YAML
- final `Conv(c1, c2, 3, 1)` for fused-feature refinement

`Align` is a small channel-alignment wrapper for future YAML variants. The current SA-DWPN-B YAML uses standard `Conv` for explicit alignment.

## File Placement In Ultralytics Source

Copy:

```text
custom_modules/sa_dwpn.py
```

to:

```text
ultralytics/nn/modules/sa_dwpn.py
```

Copy:

```text
experiments/yolo11n_sa_dwpn_b.yaml
```

to:

```text
ultralytics/cfg/models/11/yolo11n-sa-dwpn.yaml
```

## Module Registration

In `ultralytics/nn/modules/__init__.py`, add:

```python
from .sa_dwpn import Align, DWDown, SDWF
```

In `ultralytics/nn/tasks.py`, ensure these names are imported:

```python
from ultralytics.nn.modules import Align, DWDown, SDWF
```

## parse_model Changes

The exact `parse_model()` layout depends on your Ultralytics version. Keep the edit minimal.

If your version uses `base_modules`, add `Align`, `DWDown`, and `SDWF` to `base_modules`. Do not add `SDWF` to `repeat_modules`.

For `SDWF`, the YAML uses:

```yaml
- [[13, 20, 12], 1, SDWF, [256, 3, True, False]]
```

where the first argument is `c2`. `parse_model()` should inject `c1` from the first input branch and scale `c2` with the same width logic used by `Conv`:

```python
elif m is SDWF:
    c1 = ch[f[0]] if isinstance(f, list) else ch[f]
    c2 = args[0]
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
```

For `DWDown` and `Align`, ordinary Conv-like parsing is sufficient:

```python
elif m in {Align, DWDown}:
    c1, c2 = ch[f], args[0]
    if c2 != nc:
        c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
```

After each custom module is parsed, ensure `c2` is appended to the channel list for later layer indexing.

## Required Checks Before Training

Run these in order:

```bash
python tools/test_sa_dwpn_build.py --model ultralytics/cfg/models/11/yolo11n-sa-dwpn.yaml
python tools/test_sa_dwpn_weight_transfer.py --model ultralytics/cfg/models/11/yolo11n-sa-dwpn.yaml --weights yolo11n.pt
python tools/train_sa_dwpn_smoke.py --model ultralytics/cfg/models/11/yolo11n-sa-dwpn.yaml --weights yolo11n.pt --data /content/drive/MyDrive/ship_detection/data/data.yaml --project /content/drive/MyDrive/ship_detection/runs
```

Do not start full training until build, forward, and weight-transfer checks pass.

## Design Choices

SA-DWPN-B uses the simpler two-input O5 fusion:

```text
O5 = SDWF([L5, Down(O4)])
```

The task document allowed this simpler version. It avoids duplicating `L5` only to force a three-input block and keeps the first-stage experiment easier to interpret.

Spatial gating is implemented in the module but disabled in the YAML to avoid mixing too many variables into the first ablation.
