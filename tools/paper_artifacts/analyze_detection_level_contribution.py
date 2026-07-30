"""Measure real pre-NMS classification candidates contributed by each Detect level."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from custom_modules.register import register_custom_modules
    from ultralytics import YOLO

    register_custom_modules()
    model = YOLO(args.weights)
    network = model.model.eval()
    detect = network.model[-1]
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.clear()
        captured.extend(feature.detach().clone() for feature in inputs[0])

    handle = detect.register_forward_pre_hook(capture)
    paths = [
        item
        for value in args.images
        for item in (
            sorted(Path(value).rglob("*"))
            if Path(value).is_dir()
            else [Path(value)]
        )
        if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    ]
    if args.limit > 0:
        paths = paths[: args.limit]
    rows = []
    try:
        for path in paths:
            model.predict(
                source=str(path),
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )
            if len(captured) != len(detect.stride):
                raise RuntimeError("Detect hook did not capture every input level.")
            with torch.inference_mode():
                for branch, (feature, stride) in enumerate(
                    zip(captured, detect.stride, strict=True)
                ):
                    logits = detect.cv3[branch](feature)
                    score = logits.sigmoid().amax(dim=1)
                    level = f"P{int(round(torch.log2(stride).item()))}"
                    rows.append(
                        {
                            "image": path.name,
                            "branch": branch,
                            "level": level,
                            "stride": float(stride),
                            "height": int(score.shape[-2]),
                            "width": int(score.shape[-1]),
                            "candidate_locations": int(score.numel()),
                            "candidates_above_conf": int(
                                (score >= args.conf).sum().item()
                            ),
                            "classification_score_sum": float(score.sum()),
                            "classification_score_mean": float(score.mean()),
                            "scope": "pre-NMS per-location maximum class score",
                        }
                    )
    finally:
        handle.remove()
    totals: dict[str, float] = {}
    for row in rows:
        totals[row["image"]] = totals.get(row["image"], 0.0) + float(
            row["classification_score_sum"]
        )
    for row in rows:
        total = totals[row["image"]]
        row["score_mass_share"] = (
            float(row["classification_score_sum"]) / total if total else 0.0
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
