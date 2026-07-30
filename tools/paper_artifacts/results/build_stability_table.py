"""Build mean/std stability rows from completed multi-seed manifests."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from tools.paper_artifacts.results.common import (
    collect_metrics,
    placeholder,
    write_rows,
)


def build_stability(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for run_id in ("R00", "R02", "R10"):
        by_seed = [
            collect_metrics(root, run_ids=[run_id], seed=seed)[0]
            for seed in (0, 1, 2)
        ]
        row: dict[str, Any] = {
            "run_id": run_id,
            "completed_seeds": sum(item["status"] == "completed" for item in by_seed),
            "required_seeds": 3,
        }
        for metric in ("recall", "map50", "map50_95"):
            values = [
                float(item[metric])
                for item in by_seed
                if isinstance(item[metric], (int, float))
            ]
            row[f"{metric}_mean"] = (
                statistics.fmean(values)
                if len(values) == 3
                else placeholder(run_id, f"{metric}_mean")
            )
            row[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) == 3
                else placeholder(run_id, f"{metric}_std")
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--output", default="paper_artifacts/stability")
    args = parser.parse_args()
    rows = build_stability(args.run_root)
    print(
        write_rows(
            rows,
            Path(args.output),
            title="Multi-seed stability",
        )
    )


if __name__ == "__main__":
    main()
