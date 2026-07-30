"""Build objective D0/D1/D2 per-image comparison rankings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_artifacts.results.common import write_rows


def _records(path: str | Path) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {record["image"]: record for record in payload["records"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", required=True)
    parser.add_argument("--d1", required=True)
    parser.add_argument("--d2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = {"D0": _records(args.d0), "D1": _records(args.d1), "D2": _records(args.d2)}
    common = sorted(set.intersection(*(set(value) for value in inputs.values())))
    rows = []
    for image in common:
        row = {"image": image}
        for name, records in inputs.items():
            record = records[image]
            for field in ("tp", "fp", "fn", "precision", "recall"):
                row[f"{name}_{field}"] = record[field]
        row["D1_error_delta_vs_D0"] = (
            row["D1_fp"] + row["D1_fn"] - row["D0_fp"] - row["D0_fn"]
        )
        row["D2_error_delta_vs_D0"] = (
            row["D2_fp"] + row["D2_fn"] - row["D0_fp"] - row["D0_fn"]
        )
        rows.append(row)
    rows.sort(key=lambda row: (row["D2_error_delta_vs_D0"], row["image"]))
    write_rows(rows, Path(args.output), title="DPLS per-image comparison")


if __name__ == "__main__":
    main()
