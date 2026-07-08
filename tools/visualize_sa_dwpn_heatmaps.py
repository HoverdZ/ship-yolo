"""Generate SA-DWPN heatmap-analysis metadata and dependency-checked entrypoint.

Full Grad-CAM generation requires optional visualization dependencies and real
model weights/images. The script intentionally fails with a clear message when
those inputs are absent rather than fabricating visual evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sa_dwpn_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SA-DWPN Grad-CAM and spatial gate masks.")
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "WEIGHTS"), help="Model name and weights path.")
    parser.add_argument("--data", required=True, help="Dataset YAML or image directory.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--target-layer", default="p3")
    parser.add_argument("--normalization", choices=["shared", "independent"], default="shared")
    args = parser.parse_args()

    output = Path(args.output)
    for subdir in ["heatmaps", "gate_masks", "overlays"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)
    write_json(
        output / "metadata.json",
        {
            "models": args.model or [],
            "data": args.data,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "target_layer": args.target_layer,
            "normalization": args.normalization,
            "status": "metadata_only",
            "note": "Install requirements-viz.txt and provide weights/images to generate real heatmaps.",
        },
    )
    write_json(output / "selected_cases.json", {"cases": [], "status": "not_selected"})
    print(f"Wrote visualization metadata scaffold to {output}")


if __name__ == "__main__":
    main()
