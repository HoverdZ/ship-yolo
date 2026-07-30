"""Verify every artifact_checksums.sha256 below a collected version."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.windows_collection import verify_checksum_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    manifests = sorted(root.rglob("artifact_checksums.sha256"))
    rows = [
        row
        for manifest in manifests
        for row in verify_checksum_manifest(manifest)
    ]
    report = {
        "root": str(root),
        "manifests": len(manifests),
        "files": len(rows),
        "failures": [row for row in rows if not row["passed"]],
        "passed": bool(manifests) and all(row["passed"] for row in rows),
    }
    output = Path(args.output) if args.output else root / "verification_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise SystemExit("Artifact verification failed; inspect the report.")
    print(output)


if __name__ == "__main__":
    main()
