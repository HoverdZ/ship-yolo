# Deterministic ship-scale analysis

This package converts frozen YOLO horizontal boxes into paper-ready scale
statistics at a deterministic 640-pixel letterbox input. It never trains a
model, runs YOLO validation, repairs labels, changes a split, or writes to the
dataset.

## Central configuration

```python
DATA_YAML = "path/to/frozen/data.yaml"
TARGET_IMGSZ = 640
OUTPUT_DIR = "analysis/ship_scale"
DATASET_ROOT = None  # optional read-only mirror when YAML paths belong to Colab
```

Run from the repository root:

```bash
python -m tools.paper_artifacts.ship_scale_analysis \
  --data-yaml "path/to/frozen/data.yaml" \
  --imgsz 640 \
  --output-dir analysis/ship_scale
```

For a local read-only mirror of a Drive dataset whose YAML contains a
historical `/content/...` root:

```powershell
python -m tools.paper_artifacts.ship_scale_analysis `
  --data-yaml "path\to\downloaded\data.yaml" `
  --dataset-root "D:\" `
  --imgsz 640 `
  --output-dir analysis\ship_scale `
  --source-label "ship_detection/data/data.yaml" `
  --publish-root "D:\遥感船舶检测论文"
```

The public artifact records only `source-label` and dataset-root-relative
paths. It never records `--dataset-root`.

Existing output is rejected. Use `--overwrite` only after checking the exact
target. If `--publish-root` is supplied, the result is copied to a new
versioned `数据集尺度分析` subdirectory and no existing version is overwritten.
Without a publish root, a deterministic `ship_scale_analysis_bundle.zip` is
created beside the output directory.

## Input support

Each `train`, `val`, or `test` entry may be:

- an image directory;
- a UTF-8 text list;
- a relative path;
- an absolute path;
- a list containing any of the above.

Image size is read from each source image. A label path is resolved by
replacing the last `images` path component with `labels` and changing the
suffix to `.txt`. Unreadable images, missing labels, malformed rows,
out-of-range normalized coordinates, and duplicate image references are
reported in `raw_tables/reported_source_issues.csv`; no repair is attempted.

## Geometry

For original dimensions `(W, H)`:

```text
r = min(640 / W, 640 / H)
w_input = r * width_norm * W
h_input = r * height_norm * H
short = min(w_input, h_input)
long = max(w_input, h_input)
area = w_input * h_input
aspect = long / max(short, epsilon)
```

Letterbox padding does not change box width or height. These are horizontal
annotation-box dimensions, not a rotated physical hull width.

Quantiles use:

```python
numpy.quantile(values, probabilities, method="linear")
```

The spatial dilution rate is:

```text
max(0, 1 - short_side_quantile / stride) * 100
```

## Output

```text
analysis/ship_scale/
├── raw_tables/
├── paper_tables/
├── figures/
├── reports/
├── manuscript_snippets/
├── checksums/
├── artifact_manifest.json
└── artifact_checksums.sha256
```

PNG files are saved at 300 dpi and every figure also has a vector PDF.
Plotting uses Matplotlib with the non-interactive `Agg` backend and does not
use Seaborn. CSV files retain full precision; paper-facing Markdown and LaTeX
tables use two decimals.

Parquet is written only when `pyarrow` or `fastparquet` is installed. Otherwise
`raw_tables/parquet_unavailable.json` explains the optional dependency.

## Validation

```bash
python -m tools.paper_artifacts.ship_scale_analysis.validate_scale_analysis analysis/ship_scale
pytest -q tests/test_ship_scale_analysis.py
```

Validation independently recomputes quantiles and dilution rates, reconciles
split counts, checks instance uniqueness and plot-data parity, verifies
CSV/JSON agreement, and verifies every SHA-256 checksum.
