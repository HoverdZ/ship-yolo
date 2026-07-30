"""Safely extract a Colab export ZIP into a versioned Windows folder."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.windows_collection import destination_version, sha256, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("destination_root")
    parser.add_argument("--version")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    archive_path = Path(args.zip_path).expanduser().resolve()
    destination = destination_version(
        args.destination_root,
        args.version,
        allow_resume=args.resume,
    )
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            if target.is_file():
                if target.stat().st_size == member.file_size and sha256(target) == expected:
                    rows.append(
                        {
                            "source": f"{archive_path}!{member.filename}",
                            "destination": str(target),
                            "bytes": member.file_size,
                            "sha256": expected,
                            "status": "verified_skip",
                        }
                    )
                    continue
                raise FileExistsError(
                    f"Refusing to overwrite a different file: {target}"
                )
            temporary = target.with_name(target.name + ".partial")
            with archive.open(member) as source, temporary.open("wb") as stream:
                while block := source.read(1024 * 1024):
                    stream.write(block)
            if sha256(temporary) != expected:
                raise IOError(f"SHA256 mismatch extracting {member.filename}")
            os.replace(temporary, target)
            rows.append(
                {
                    "source": f"{archive_path}!{member.filename}",
                    "destination": str(target),
                    "bytes": member.file_size,
                    "sha256": expected,
                    "status": "copied",
                }
            )
    print(write_report(destination, rows, source=str(archive_path)))


if __name__ == "__main__":
    main()
