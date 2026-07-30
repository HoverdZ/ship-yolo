"""Copy a Drive-for-Desktop formal project tree into a versioned Windows folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.windows_collection import (
    copy_tree,
    destination_version,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination_root")
    parser.add_argument("--version")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    destination = destination_version(
        args.destination_root,
        args.version,
        allow_resume=args.resume,
    )
    rows = copy_tree(args.source, destination)
    print(write_report(destination, rows, source=args.source))


if __name__ == "__main__":
    main()
