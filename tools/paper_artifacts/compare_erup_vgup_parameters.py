"""Recompute ERUP/VGUP parameter counts from current formal code."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_modules.erup import ERUPPreprocessor
from custom_modules.vgup import VGUPPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    modules = {
        "ERUP": ERUPPreprocessor(),
        "VGUP": VGUPPreprocessor(),
    }
    counts = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in modules.items()
    }
    rows = [
        {"module": name, "parameters": value}
        for name, value in counts.items()
    ]
    rows.append(
        {
            "module": "VGUP/ERUP",
            "parameters": counts["VGUP"] / counts["ERUP"],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["module", "parameters"])
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "ERUP_parameters": counts["ERUP"],
                "VGUP_parameters": counts["VGUP"],
                "VGUP_ERUP_ratio": counts["VGUP"] / counts["ERUP"],
                "VGUP_as_percent_of_ERUP": (
                    100.0 * counts["VGUP"] / counts["ERUP"]
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
