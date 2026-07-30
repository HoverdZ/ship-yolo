"""Command-line adapter for a named formal result table."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.paper_artifacts.results.builders import build


def run_builder(name: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument(
        "--output",
        default=f"paper_artifacts/{name}",
    )
    args = parser.parse_args()
    print(build(name, args.run_root, Path(args.output)))


__all__ = ["run_builder"]
