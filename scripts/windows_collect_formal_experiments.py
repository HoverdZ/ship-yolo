"""Safely import six formal Colab ZIPs into a versioned Windows directory."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

EXPERIMENT_IDS = [
    "A0_yolo11n",
    "A1_inceptiondw",
    "A2_inceptiondw_dpls",
    "A3_inceptiondw_dpls_scam",
    "A4_inceptiondw_dpls_scam_vgup",
    "A5_inceptiondw_dpls_ca_scam_vgup",
]
REQUIRED = {
    "run_manifest.json",
    "artifact_checksums.sha256",
    "best_epoch_summary.json",
    "complexity.json",
    "val_predictions.json",
    "val_image_metrics.csv",
    "weights/best.pt",
    "weights/last.pt",
}


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)


def locate(source: Path, experiment_id: str) -> Path:
    matches = sorted(source.rglob(f"{experiment_id}_paper_artifacts.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Missing package for {experiment_id} under {source}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Drive Desktop sync root or manual-download directory")
    parser.add_argument("--target", default=r"D:\遥感船舶检测论文\formal_ablation_v1")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    target_root = Path(args.target).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import_root = target_root / "imports" / timestamp
    if import_root.exists():
        raise FileExistsError(import_root)
    import_root.mkdir(parents=True)
    manifest = {"source": str(source), "target": str(import_root), "experiments": []}
    for experiment_id in EXPERIMENT_IDS:
        archive = locate(source, experiment_id)
        safe_extract(archive, import_root)
        candidates = list((import_root / experiment_id).rglob("run_manifest.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"{experiment_id}: expected one run_manifest.json, found {len(candidates)}")
        run_dir = candidates[0].parent
        present = {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()}
        missing = sorted(REQUIRED - present)
        if missing:
            raise RuntimeError(f"{experiment_id}: incomplete package, missing {missing}")
        manifest["experiments"].append({"experiment_id": experiment_id, "archive": str(archive), "run_dir": str(run_dir), "missing": []})
    (import_root / "windows_collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.skip_summary:
        repo = Path(args.repo).resolve()
        subprocess.run([sys.executable, str(repo / "tools/paper_artifacts/collect_run_artifacts.py"), str(import_root)], check=True, cwd=repo)
        subprocess.run([sys.executable, str(repo / "tools/paper_artifacts/generate_paper_tables.py"), str(import_root)], check=True, cwd=repo)
        subprocess.run([sys.executable, str(repo / "tools/paper_artifacts/select_visual_examples.py"), str(import_root)], check=True, cwd=repo)
    print(import_root)


if __name__ == "__main__":
    main()
