"""Build and audit the official AC-YOLO graph on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_YAML = ROOT / "experiments" / "paper_comparisons" / "P05_yolo11n_ac_yolo.yaml"
EXPECTED_HEAD = [
    "Conv",
    "Upsample",
    "Conv",
    "Concat",
    "C3k2",
    "Conv",
    "Upsample",
    "Conv",
    "Concat",
    "C3k2",
    "Conv",
    "Concat",
    "C3k2",
    "SCDown",
    "Concat",
    "C3k2",
    "Detect",
]


def audit(imgsz: int = 128) -> dict:
    """Return deterministic structure and CPU-gradient checks."""

    from custom_modules.ac_yolo_official import ACmix, C2PSA_ACmix
    from custom_modules.register import register_custom_modules

    register_custom_modules()
    from ultralytics import YOLO

    core = YOLO(str(MODEL_YAML), task="detect").model.cpu().train()
    sample = torch.randn(1, 3, imgsz, imgsz, requires_grad=True)
    prediction = core(sample)
    loss = prediction["boxes"].float().square().mean()
    loss.backward()
    head = [type(core.model[index]).__name__ for index in range(11, 28)]
    report = {
        "model_yaml": str(MODEL_YAML.relative_to(ROOT)),
        "official_source_commit": core.yaml.get("official_source_commit"),
        "parameters": sum(parameter.numel() for parameter in core.parameters()),
        "strides": [int(value) for value in core.stride.tolist()],
        "c2psa_acmix_count": sum(
            isinstance(module, C2PSA_ACmix) for module in core.modules()
        ),
        "acmix_count": sum(isinstance(module, ACmix) for module in core.modules()),
        "ccfm_head_types": head,
        "finite_input_gradient": bool(
            sample.grad is not None and torch.isfinite(sample.grad).all()
        ),
    }
    report["passed"] = bool(
        report["official_source_commit"]
        == "20dad8db5047add008e6eab65b032158f4a5d3e1"
        and report["strides"] == [8, 16, 32]
        and report["c2psa_acmix_count"] == 1
        and report["acmix_count"] >= 1
        and report["ccfm_head_types"] == EXPECTED_HEAD
        and report["finite_input_gradient"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(imgsz=args.imgsz)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("AC-YOLO structure audit failed.")


if __name__ == "__main__":
    main()
