"""Collect canonical metrics from real run manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.paper_artifacts.results.common import collect_metrics, write_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--output", default="paper_artifacts/all_run_metrics")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = collect_metrics(args.run_root, seed=args.seed)
    paths = write_rows(
        rows,
        Path(args.output),
        title="Canonical formal-run metrics",
    )
    print(paths)


if __name__ == "__main__":
    main()
