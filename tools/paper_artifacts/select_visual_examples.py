"""Select comparison images from deterministic TP/FP/FN deltas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_artifacts.compare_per_image_predictions import compare


def select(root: str | Path, output: str | Path | None = None, count: int = 5) -> pd.DataFrame:
    frame = compare(root)
    selections = []
    rules = {
        "DPLS_reduces_misses": ("A2_delta_fn_vs_A0", True),
        "VGUP_reduces_misses": ("A4_delta_fn_vs_A0", True),
        "VGUP_reduces_false_positives": ("A4_delta_fp_vs_A0", True),
        "CA_SCAM_vs_SCAM_reduces_misses": (None, True),
        "CA_SCAM_vs_SCAM_reduces_false_positives": (None, True),
        "CA_SCAM_regression": (None, False),
    }
    for rule, (column, ascending) in rules.items():
        ranked = frame.copy()
        if rule == "CA_SCAM_vs_SCAM_reduces_misses":
            ranked["score"] = ranked["A5_fn"] - ranked["A4_fn"]
            ascending = True
        elif rule == "CA_SCAM_vs_SCAM_reduces_false_positives":
            ranked["score"] = ranked["A5_fp"] - ranked["A4_fp"]
            ascending = True
        elif rule == "CA_SCAM_regression":
            ranked["score"] = (ranked["A5_fn"] + ranked["A5_fp"]) - (ranked["A4_fn"] + ranked["A4_fp"])
            ascending = False
        else:
            ranked["score"] = ranked[column]
        for rank, (_, row) in enumerate(ranked.sort_values(["score", "image"], ascending=[ascending, True]).head(count).iterrows(), start=1):
            selections.append({"rule": rule, "rank": rank, "image": row["image"], "score": row["score"], "selection_basis": "validation per-image error delta; test split unused"})
    result = pd.DataFrame(selections)
    destination = Path(output) if output else Path(root) / "paper_summary" / "visual_selection_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    print(select(args.root, args.output, args.count).to_string(index=False))


if __name__ == "__main__":
    main()
