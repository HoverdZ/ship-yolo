"""Generate bounded common and module-specific visual evidence after training."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from tools.paper_artifacts.formal_protocol import FormalConfig, write_json
from tools.paper_artifacts.generate_heatmaps import generate_one


def _load(path: str, imgsz: int, device: torch.device) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB")
    resized = image.resize((imgsz, imgsz), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, image


def _save_scalar_map(value: torch.Tensor, output: Path) -> None:
    array = value.detach().float().squeeze().cpu().numpy()
    array -= array.min()
    array /= max(float(array.max()), 1e-8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(array * 255)).save(output)


def _save_feature_energy(value: torch.Tensor, output: Path) -> None:
    _save_scalar_map(value.detach().float().square().mean(dim=1, keepdim=True), output)


def _draw(record: dict[str, Any], output: Path, mode: str) -> None:
    image = Image.open(record["source_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    if mode in {"gt", "both"}:
        for truth in record["ground_truth"]:
            draw.rectangle(truth["xyxy"], outline=(30, 220, 30), width=3)
    if mode in {"prediction", "both"}:
        for prediction in record["predictions"]:
            draw.rectangle(prediction["xyxy"], outline=(230, 40, 40), width=3)
            draw.text((prediction["xyxy"][0], prediction["xyxy"][1]), f"{prediction['confidence']:.2f}", fill=(230, 40, 40))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _select(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    categories = {
        "miss": sorted(records, key=lambda item: (-item["fn"], item["image"])),
        "false_positive": sorted(records, key=lambda item: (-item["fp"], item["image"])),
        "high_confidence": sorted(records, key=lambda item: (-max([p["confidence"] for p in item["predictions"]] or [0]), item["image"])),
        "difficult": sorted([item for item in records if item["gt_count"]], key=lambda item: (item["recall"], item["precision"], item["image"])),
        "negative": sorted([item for item in records if item["gt_count"] == 0], key=lambda item: (-item["prediction_count"], item["image"])),
    }
    selected = []
    seen = set()
    for category, candidates in categories.items():
        if candidates:
            record = candidates[0]
            selected.append((category, record))
            seen.add(record["image"])
    for record in records:
        if len(selected) >= 6:
            break
        if record["image"] not in seen:
            selected.append(("representative", record))
            seen.add(record["image"])
    return selected


def _feature_stats(model, selected: list[tuple[str, dict[str, Any]]], layer_indices: list[int], output_dir: Path, imgsz: int) -> list[dict[str, Any]]:
    network = model.model.eval()
    device = next(network.parameters()).device
    rows = []
    for category, record in selected[:4]:
        activations: dict[int, torch.Tensor] = {}
        handles = [
            network.model[index].register_forward_hook(lambda _m, _a, out, idx=index: activations.__setitem__(idx, out.detach() if isinstance(out, torch.Tensor) else out[0].detach()))
            for index in layer_indices
        ]
        tensor, _ = _load(record["source_path"], imgsz, device)
        try:
            with torch.inference_mode():
                network(tensor)
        finally:
            for handle in handles:
                handle.remove()
        for index, activation in activations.items():
            rows.append({
                "image": record["image"],
                "category": category,
                "layer": index,
                "module": type(network.model[index]).__name__,
                "mean_abs": float(activation.abs().mean()),
                "std": float(activation.std()),
                "energy": float(activation.square().mean()),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "feature_energy_statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["image", "category", "layer", "module", "mean_abs", "std", "energy"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _vgup_stats(config: FormalConfig, model, records: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    network = model.model.eval()
    preprocessor = network.model[0]
    device = next(network.parameters()).device
    rows = []
    visual_saved = False
    for record in records:
        tensor, original = _load(record["source_path"], config.imgsz, device)
        with torch.inference_mode():
            _output, debug = preprocessor(tensor, return_debug=True)
        gate = debug["spatial_gate"]
        row = {
            "image": record["image"],
            "global_acceptance_gate": float(debug["global_gate"].mean()),
            "spatial_gate_mean": float(gate.mean()),
            "spatial_gate_std": float(gate.std()),
            "spatial_gate_min": float(gate.min()),
            "spatial_gate_max": float(gate.max()),
            "bpw_parameter_mean": float(debug["bpw_params"].mean()),
            "bpw_parameter_std": float(debug["bpw_params"].std()),
            "kbl_parameter_mean": float(debug["kbl_params"].mean()),
            "kbl_parameter_std": float(debug["kbl_params"].std()),
            "input_luminance_std": float(tensor.mean(dim=1).std()),
            "output_luminance_std": float(debug["output_image"].mean(dim=1).std()),
        }
        rows.append(row)
        if not visual_saved:
            luminance: dict[str, np.ndarray] = {}
            for name, value in {
                "original": tensor,
                "bpw": debug["bpw_image"],
                "gated_bpw": debug["gated_bpw_image"],
                "kbl": debug["kbl_image"],
                "vgup_output": debug["output_image"],
            }.items():
                array = value[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                Image.fromarray(np.uint8(array * 255)).save(output_dir / f"{name}.png")
                luminance[name] = array.mean(axis=2)
            gate_array = debug["spatial_gate"][0, 0].detach().cpu().numpy()
            Image.fromarray(np.uint8(gate_array * 255)).save(output_dir / "vgup_spatial_visibility_gate.png")
            try:
                import matplotlib.pyplot as plt

                figure, axis = plt.subplots(figsize=(7, 4))
                for name in ("original", "bpw", "gated_bpw", "kbl", "vgup_output"):
                    axis.hist(luminance[name].ravel(), bins=64, range=(0, 1), density=True, histtype="step", label=name)
                axis.set(xlabel="Normalized luminance", ylabel="Density", title="VGUP luminance distribution")
                axis.legend(fontsize=7)
                figure.tight_layout()
                figure.savefig(output_dir / "vgup_luminance_histogram.png", dpi=300)
                plt.close(figure)
            except Exception as error:
                (output_dir / "vgup_luminance_histogram.failed.txt").write_text(
                    f"{type(error).__name__}: {error}\n",
                    encoding="utf-8",
                )
            visual_saved = True
    with (config.run_dir / "vgup_gate_statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _scam_stats(config: FormalConfig, model, records: list[dict[str, Any]], output_dir: Path, calibrated: bool) -> list[dict[str, Any]]:
    network = model.model.eval()
    device = next(network.parameters()).device
    modules = [(index, layer) for index, layer in enumerate(network.model) if type(layer).__name__ in ({"CASCAM"} if calibrated else {"SCAM"})]
    rows = []
    maps_saved = False
    for record in records:
        tensor, _ = _load(record["source_path"], config.imgsz, device)
        captured: dict[int, torch.Tensor] = {}
        handles = [layer.register_forward_pre_hook(lambda _m, args, idx=index: captured.__setitem__(idx, args[0].detach())) for index, layer in modules]
        try:
            with torch.inference_mode():
                network(tensor)
        finally:
            for handle in handles:
                handle.remove()
        for level, (index, layer) in enumerate(modules, start=2):
            feature = captured[index]
            with torch.inference_mode():
                residual = layer.compute_context_residual(feature)
                row = {
                    "image": record["image"],
                    "level": f"P{level}",
                    "input_energy": float(feature.square().mean()),
                    "residual_before_mean": float(residual.abs().mean()),
                    "output_change_ratio": float(residual.abs().mean() / feature.abs().mean().clamp_min(1e-8)),
                }
                if calibrated:
                    local_contrast, contrast_map, beta = layer.contrast_state(feature)
                    residual_after = (1 + beta * contrast_map) * residual
                    row.update({
                        "beta": float(beta),
                        "contrast_mean": float(local_contrast.mean()),
                        "contrast_std": float(local_contrast.std()),
                        "contrast_gate_mean": float(contrast_map.mean()),
                        "residual_after_mean": float(residual_after.abs().mean()),
                    })
                    if not maps_saved:
                        _save_scalar_map(local_contrast, output_dir / f"P{level}_local_contrast.png")
                        _save_scalar_map(contrast_map, output_dir / f"P{level}_contrast_gate.png")
                        _save_feature_energy(residual_after, output_dir / f"P{level}_calibrated_residual.png")
                if not maps_saved:
                    _save_feature_energy(feature, output_dir / f"P{level}_scam_input_energy.png")
                    _save_feature_energy(residual, output_dir / f"P{level}_context_residual.png")
                    output_feature = feature + (
                        ((1 + beta * contrast_map) * residual) if calibrated else residual
                    )
                    _save_feature_energy(output_feature, output_dir / f"P{level}_scam_output_energy.png")
            rows.append(row)
        maps_saved = True
    filename = "ca_scam_statistics.csv" if calibrated else "scam_statistics.csv"
    with (config.run_dir / filename).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["image", "level"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def generate_specialty_artifacts(config: FormalConfig, model, predictions: dict[str, Any]) -> dict[str, Any]:
    records = predictions["records"]
    selected = _select(records)
    root = config.run_dir / "visualizations"
    for directory in ("detections", "errors", "heatmaps", "module"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for category, record in selected:
        safe = Path(record["image"]).stem
        _draw(record, root / "detections" / f"{category}_{safe}_gt.png", "gt")
        _draw(record, root / "detections" / f"{category}_{safe}_prediction.png", "prediction")
        if category in {"miss", "false_positive", "difficult"}:
            _draw(record, root / "errors" / f"{category}_{safe}.png", "both")
        manifest_rows.append({"category": category, "image": record["image"], "reason": f"deterministic ranking by {category} criterion", "fn": record["fn"], "fp": record["fp"], "precision": record["precision"], "recall": record["recall"]})
    with (config.run_dir / "visual_selection_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]) if manifest_rows else ["category", "image", "reason"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    network = model.model
    detect_from = config.spec["detect_from"]
    specialty = config.spec["specialty"]
    specialty_layers = {
        "baseline": detect_from,
        "inceptiondw": [2, 4],
        "dpls": detect_from,
        "scam": [14, 17, 20, *detect_from],
        "vgup": [15, 18, 21, *detect_from],
        "ca_scam": [15, 18, 21, *detect_from],
    }[specialty]
    if selected:
        Image.open(selected[0][1]["source_path"]).convert("RGB").save(root / "module" / "input_reference.png")
    feature_rows = _feature_stats(model, selected, specialty_layers, root / "module", config.imgsz)
    heatmap_reports = []
    if selected:
        for index in specialty_layers:
            heatmap_reports.append(
                generate_one(
                    model,
                    selected[0][1]["source_path"],
                    index,
                    root / "heatmaps" / f"layer_{index}_feature_energy.png",
                    "feature-energy",
                    config.imgsz,
                )
            )
        heatmap_reports.append(
            generate_one(
                model,
                selected[0][1]["source_path"],
                detect_from[0],
                root / "heatmaps" / f"layer_{detect_from[0]}_gradcam.png",
                "gradcam",
                config.imgsz,
            )
        )
    pyramid_rows = []
    for index, stride in zip(detect_from, config.spec["strides"], strict=True):
        pyramid_rows.append({
            "level": f"P{int(round(torch.log2(torch.tensor(stride)).item()))}",
            "layer": index,
            "stride": stride,
            "grid_height": config.imgsz // int(stride),
            "grid_width": config.imgsz // int(stride),
            "candidate_locations": (config.imgsz // int(stride)) ** 2,
            "selected_boxes_total_all_levels": sum(record["prediction_count"] for record in records),
            "level_assignment_note": "Ultralytics NMS output does not retain source-level IDs; candidate grid contribution is reported without fabricating per-level NMS attribution.",
        })
    with (config.run_dir / "pyramid_level_statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pyramid_rows[0]))
        writer.writeheader()
        writer.writerows(pyramid_rows)

    module_rows: list[dict[str, Any]] = []
    vgup_rows: list[dict[str, Any]] = []
    if specialty in {"scam", "vgup"}:
        module_rows = _scam_stats(config, model, records, root / "module", calibrated=False)
    if specialty in {"vgup", "ca_scam"}:
        vgup_rows = _vgup_stats(config, model, records, root / "module")
    if specialty == "ca_scam":
        module_rows = _scam_stats(config, model, records, root / "module", calibrated=True)
    report = {
        "specialty": specialty,
        "selected_images": len(selected),
        "feature_stat_rows": len(feature_rows),
        "module_stat_rows": len(module_rows),
        "vgup_gate_rows": len(vgup_rows),
        "heatmaps": heatmap_reports,
        "heatmap_failures_are_non_blocking": True,
        "large_intermediate_tensors_saved": False,
    }
    write_json(config.run_dir / "specialty_artifacts_report.json", report)
    return report


__all__ = ["generate_specialty_artifacts"]
