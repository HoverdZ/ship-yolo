"""Create a complete, checksummed formal-experiment ZIP without deleting data."""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path


def export_bundle(run_dir: str | Path, destination: str | Path) -> Path:
    source = Path(run_dir).resolve()
    output = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(item for item in source.rglob("*") if item.is_file()):
                archive.write(path, arcname=f"{source.name}/{path.relative_to(source).as_posix()}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(export_bundle(args.run_dir, args.destination))


if __name__ == "__main__":
    main()
