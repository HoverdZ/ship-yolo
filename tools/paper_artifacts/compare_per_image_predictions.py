"""Join image metrics across runs and calculate objective error deltas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_artifacts.formal_protocol import EXPERIMENTS


def compare(root: str | Path, output: str | Path | None = None) -> pd.DataFrame:
    base = Path(root)
    frames = []
    for experiment_id in EXPERIMENTS:
        frame = pd.read_csv(base / experiment_id / "val_image_metrics.csv")
        prefix = experiment_id.split("_")[0]
        frame = frame[["image", "tp", "fp", "fn", "precision", "recall"]].rename(columns={column: f"{prefix}_{column}" for column in ("tp", "fp", "fn", "precision", "recall")})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="image", how="outer", validate="one_to_one")
    for prefix in ("A1", "A2", "A3", "A4", "A5"):
        merged[f"{prefix}_delta_fn_vs_A0"] = merged[f"{prefix}_fn"] - merged["A0_fn"]
        merged[f"{prefix}_delta_fp_vs_A0"] = merged[f"{prefix}_fp"] - merged["A0_fp"]
    destination = Path(output) if output else base / "paper_summary" / "per_image_comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(f"Compared {len(compare(args.root, args.output))} validation images")


if __name__ == "__main__":
    main()
