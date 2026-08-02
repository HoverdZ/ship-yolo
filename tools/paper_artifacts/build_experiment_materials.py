"""Build the Windows CPU portion of the Ocean Engineering evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_artifacts.collect_training_logs import build_training_materials
from tools.paper_artifacts.compute_model_complexity import compute_complexities
from tools.paper_artifacts.extract_weight_metadata import (
    audit_experiment_results,
    sha256_file,
)

CATEGORY_DIRS = {
    "总体": "00_总体材料",
    "DPLS": "01_DPLS",
    "CA-SCAM": "02_CA-SCAM",
    "VGUP": "03_VGUP",
    "浅层卷积": "04_浅层卷积算子探索",
}

DOCUMENT_REPORTED_METRICS = {
    "YOLO11n": (0.838, 0.718, 0.786, 0.326),
    "YOLO11n + InceptionDW": (0.841, 0.716, 0.788, 0.332),
    "YOLO11n + PConv": (0.844, 0.692, 0.783, 0.320),
    "YOLO11n + LSKConv": (0.805, 0.733, 0.783, 0.329),
    "YOLO11n + InceptionDW + DPLS": (0.826, 0.737, 0.800, 0.336),
    "YOLO11n + InceptionDW + DPLS + SCAM": (0.840, 0.710, 0.797, 0.339),
    "YOLO11n + InceptionDW + DPLS + SCAM + VGUP": (0.824, 0.729, 0.797, 0.338),
    "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP": (0.837, 0.728, 0.804, 0.344),
    "YOLO11n + InceptionDW + DPLS + CA-SCAM": (0.818, 0.736, 0.809, 0.337),
    "YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP": (0.811, 0.731, 0.793, 0.331),
    "YOLO11n + PLS + CA-SCAM + VGUP": (0.796, 0.725, 0.790, 0.332),
}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or (list(rows[0]) if rows else ["状态"])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _prepare_directories(material_root: Path) -> dict[str, Path]:
    material_root.mkdir(parents=True, exist_ok=True)
    directories = {}
    for key, name in CATEGORY_DIRS.items():
        path = material_root / name
        path.mkdir(exist_ok=True)
        directories[key] = path
    for path in directories.values():
        nested = [item for item in path.iterdir() if item.is_dir()]
        if nested:
            raise ValueError(f"Material folder must stay flat: {path}; nested={nested}")
    return directories


def _git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _write_environment_audit(output: Path, repo: Path, audit: dict[str, Any]) -> None:
    try:
        import ultralytics
        import thop

        local_ultralytics = ultralytics.__version__
        thop_version = getattr(thop, "__version__", "unknown")
    except Exception as error:
        local_ultralytics = f"unavailable: {error}"
        thop_version = "unavailable"
    training_versions = sorted(
        {str(record.get("training_version")) for record in audit["records"]}
    )
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "local_ultralytics": local_ultralytics,
        "thop": thop_version,
        "checkpoint_training_versions": training_versions,
        "git_branch": _git_value(repo, "branch", "--show-current"),
        "git_commit": _git_value(repo, "rev-parse", "HEAD"),
        "git_status": _git_value(repo, "status", "--short"),
    }
    (output / "本地环境与版本审计.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    warning = ""
    if training_versions != [str(local_ultralytics)]:
        warning = (
            "\n- 注意：checkpoint记录的训练Ultralytics版本与当前Windows审计版本不同。"
            "表格同时保存两者，GPU复核Notebook优先使用checkpoint训练版本。\n"
        )
    (output / "本地环境与版本审计.md").write_text(
        "# 本地环境与版本审计\n\n"
        f"- Python：{payload['python']}\n"
        f"- Torch：{payload['torch']}\n"
        f"- CUDA可用：{payload['cuda_available']}\n"
        f"- 当前Ultralytics：{payload['local_ultralytics']}\n"
        f"- checkpoint训练版本：{', '.join(training_versions)}\n"
        f"- THOP：{payload['thop']}\n"
        f"- Git分支：{payload['git_branch']}\n"
        f"- Git提交：{payload['git_commit']}\n"
        + warning,
        encoding="utf-8",
    )


def _write_document_metric_reconciliation(
    output: Path,
    metric_rows: list[dict[str, Any]],
) -> None:
    by_name = {row["模型"]: row for row in metric_rows}
    rows = []
    for model, reported in DOCUMENT_REPORTED_METRICS.items():
        evidence = by_name.get(model)
        for metric, reported_value in zip(("P", "R", "mAP50", "mAP50-95"), reported, strict=True):
            evidence_value = evidence.get(metric) if evidence else None
            rows.append(
                {
                    "模型": model,
                    "指标": metric,
                    "任务文档四舍五入值": reported_value,
                    "文件证据值": evidence_value,
                    "文件值减文档值": (
                        float(evidence_value) - reported_value
                        if evidence_value not in (None, "")
                        else None
                    ),
                    "正式表采用": "文件证据值" if evidence_value not in (None, "") else "任务文档值（待人工确认）",
                }
            )
    _write_csv(output / "任务文档数值与文件证据核对.csv", rows)


def _supplemental_metrics_file(output: Path) -> Path:
    p, r, map50, map95 = DOCUMENT_REPORTED_METRICS["YOLO11n + InceptionDW"]
    path = output / "任务文档补充指标.json"
    path.write_text(
        json.dumps(
            [
                {
                    "模型": "YOLO11n + InceptionDW",
                    "P": p,
                    "R": r,
                    "mAP50": map50,
                    "mAP50-95": map95,
                    "身份状态": "待人工确认",
                    "数值来源": "本轮任务文档人工提供；桌面实验结果未找到对应best.pt/results.csv",
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _copy_scale_materials(source: Path, output: Path) -> list[Path]:
    required = {
        "表06_尺度分位数原始宽表.csv": None,
        "表02_论文空间稀释率表.csv": "DPLS_各层空间稀释率.csv",
        "图01_训练集船舶短边分布.png": "DPLS_短边尺寸分布.png",
        "图01_训练集船舶短边分布_矢量版.pdf": "DPLS_短边尺寸分布.pdf",
        "图02_训练集船舶短边累积分布.png": "DPLS_短边累积分布.png",
        "图02_训练集船舶短边累积分布_矢量版.pdf": "DPLS_短边累积分布.pdf",
        "图04_训练集船舶短边长边联合分布.png": "DPLS_短边长边二维分布.png",
        "图04_训练集船舶短边长边联合分布_矢量版.pdf": "DPLS_短边长边二维分布.pdf",
    }
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing confirmed scale materials: {missing}")
    created = []
    quantiles = pd.read_csv(source / "表06_尺度分位数原始宽表.csv")
    for metric, filename in (
        ("short_side_640_px", "DPLS_船舶短边分位数.csv"),
        ("long_side_640_px", "DPLS_船舶长边分位数.csv"),
    ):
        subset = quantiles[quantiles["Metric"] == metric]
        destination = output / filename
        subset.to_csv(destination, index=False, encoding="utf-8-sig")
        created.append(destination)
    for source_name, target_name in required.items():
        if target_name is None:
            continue
        destination = output / target_name
        shutil.copy2(source / source_name, destination)
        if destination.suffix.lower() == ".csv":
            # Preserve the confirmed values while enforcing the bundle-wide
            # Excel-compatible UTF-8 BOM contract.
            text = destination.read_text(encoding="utf-8-sig")
            destination.write_text(text, encoding="utf-8-sig")
        created.append(destination)
    summary = (source / "报告_船舶尺度分析总结.md").read_text(encoding="utf-8")
    motivation = (source / "论文文字_DPLS尺度动机_中文.md").read_text(encoding="utf-8")
    note = output / "DPLS_尺度统计说明.md"
    note.write_text(
        "# DPLS尺度统计说明\n\n"
        f"来源目录：`{source}`。原材料SHA256清单与一致性报告均已存在于来源目录。\n\n"
        "## 已确认统计摘要\n\n"
        + summary
        + "\n## 已有论文动机文字\n\n"
        + motivation
        + "\n",
        encoding="utf-8",
    )
    created.append(note)
    return created


def _find_record(audit: dict[str, Any], identity: str) -> dict[str, Any]:
    matches = [record for record in audit["records"] if record["identity"] == identity]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint for {identity!r}, found {len(matches)}")
    return matches[0]


def _metrics_lookup(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["模型"]: row for row in metric_rows}


def _write_pair_metric_table(
    path: Path,
    metric_rows: list[dict[str, Any]],
    control_name: str,
    treatment_name: str,
) -> list[dict[str, Any]]:
    lookup = _metrics_lookup(metric_rows)
    control, treatment = lookup[control_name], lookup[treatment_name]
    rows = []
    for metric in ("P", "R", "mAP50", "mAP50-95"):
        rows.append(
            {
                "指标": metric,
                "对照模型": control_name,
                "对照值": control[metric],
                "实验模型": treatment_name,
                "实验值": treatment[metric],
                "变化量": float(treatment[metric]) - float(control[metric]),
            }
        )
    _write_csv(path, rows)
    return rows


def _load_checkpoint_model(path: str | Path) -> torch.nn.Module:
    from custom_modules.register import register_custom_modules

    register_custom_modules(patch_parse_model=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("ema") or checkpoint["model"]


def _extract_ca_scam_cpu(audit: dict[str, Any], output: Path, metric_rows: list[dict[str, Any]]) -> None:
    from custom_modules.scam import SCAM

    final_name = "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"
    scam_name = "YOLO11n + InceptionDW + DPLS + SCAM + VGUP"
    record = _find_record(audit, final_name)
    model = _load_checkpoint_model(record["checkpoint"])
    modules = [
        (index, layer)
        for index, layer in enumerate(model.model)
        if type(layer).__name__ in {"CASCAM", "CASCAMFixedBeta", "CASCAMUnbounded"}
    ]
    if len(modules) != 3:
        raise ValueError(f"Expected three CA-SCAM layers, got {len(modules)}")
    beta_rows = []
    parameter_rows = []
    structure_lines = [
        "CA-SCAM结构参数",
        f"权重文件：{record['checkpoint_name']}",
        f"权重SHA256：{record['checkpoint_sha256']}",
        "结构定义：SCAM上下文残差 + 局部对比度条件的有界残差校准。",
        "输出公式：x + (1 + beta * contrast_map) * context_residual。",
        "beta公式：max_delta * tanh(contrast_logit)。",
        "",
    ]
    for level, (index, layer) in enumerate(modules, start=2):
        beta = float(layer.calibration_beta().detach().cpu())
        in_channels = int(layer.in_channels)
        contrast_logit = (
            float(layer.contrast_logit.detach().cpu())
            if hasattr(layer, "contrast_logit")
            else None
        )
        beta_rows.append(
            {
                "层级": f"P{level}",
                "模型层索引": index,
                "in_channels": in_channels,
                "max_delta": getattr(layer, "max_delta", None),
                "contrast_logit": contrast_logit,
                "beta": beta,
                "beta来源": "checkpoint静态Parameter经calibration_beta()计算",
            }
        )
        scam_parameters = sum(parameter.numel() for parameter in SCAM(in_channels).parameters())
        ca_parameters = sum(parameter.numel() for parameter in layer.parameters())
        parameter_rows.append(
            {
                "层级": f"P{level}",
                "SCAM参数量": scam_parameters,
                "CA-SCAM参数量": ca_parameters,
                "增加参数量": ca_parameters - scam_parameters,
                "增加百分比": (ca_parameters - scam_parameters) / scam_parameters * 100.0,
            }
        )
        structure_lines.append(
            f"P{level}: layer={index}, in_channels={in_channels}, "
            f"max_delta={getattr(layer, 'max_delta', None)}, "
            f"contrast_logit={contrast_logit}, beta={beta}"
        )
    _write_csv(output / "CA-SCAM各层beta参数.csv", beta_rows)
    _write_csv(output / "CA-SCAM参数量对比.csv", parameter_rows)
    (output / "CA-SCAM结构参数.txt").write_text(
        "\n".join(structure_lines) + "\n",
        encoding="utf-8",
    )
    metrics = _write_pair_metric_table(
        output / "CA-SCAM与SCAM指标对比.csv",
        metric_rows,
        scam_name,
        final_name,
    )
    changes = {row["指标"]: row["变化量"] for row in metrics}
    (output / "CA-SCAM结果分析.md").write_text(
        "# CA-SCAM结果分析\n\n"
        f"在固定 InceptionDW + DPLS + VGUP 的近邻对照下，CA-SCAM相对SCAM的变化为："
        f"P {changes['P']:+.6f}、R {changes['R']:+.6f}、mAP50 {changes['mAP50']:+.6f}、"
        f"mAP50-95 {changes['mAP50-95']:+.6f}。\n\n"
        "该表只陈述单次正式实验的真实差值；机制可视化与代表案例需运行GPU Notebook后补入。\n",
        encoding="utf-8",
    )


def _extract_vgup_cpu(
    audit: dict[str, Any],
    output: Path,
    metric_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
) -> None:
    from custom_modules.erup import ERUPPreprocessor
    from custom_modules.vgup import VGUPPreprocessor

    erup = ERUPPreprocessor()
    vgup = VGUPPreprocessor()
    erup_params = sum(parameter.numel() for parameter in erup.parameters())
    vgup_params = sum(parameter.numel() for parameter in vgup.parameters())
    ratio = vgup_params / erup_params
    reduction = 1.0 - ratio
    complexity_lookup = {row["模型"]: row for row in complexity_rows}
    no_vgup = "YOLO11n + InceptionDW + DPLS + CA-SCAM"
    with_vgup = "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"
    rows = [
        {
            "比较层级": "预处理模块",
            "ERUP参数量": erup_params,
            "VGUP参数量": vgup_params,
            "参数变化量_VGUP减ERUP": vgup_params - erup_params,
            "VGUP_ERUP比值": ratio,
            "相对ERUP减少比例": reduction,
            "备注": "从当前真实ERUPPreprocessor/VGUPPreprocessor代码重新实例化统计",
        }
    ]
    if no_vgup in complexity_lookup and with_vgup in complexity_lookup:
        left, right = complexity_lookup[no_vgup], complexity_lookup[with_vgup]
        rows.append(
            {
                "比较层级": "完整模型（VGUP加入前后）",
                "ERUP参数量": None,
                "VGUP参数量": None,
                "参数变化量_VGUP减ERUP": int(right["Params"]) - int(left["Params"]),
                "VGUP_ERUP比值": None,
                "相对ERUP减少比例": None,
                "备注": (
                    f"无VGUP Params={left['Params']}, GFLOPs={left.get('GFLOPs@640')}; "
                    f"有VGUP Params={right['Params']}, GFLOPs={right.get('GFLOPs@640')}"
                ),
            }
        )
    _write_csv(output / "VGUP与ERUP复杂度对比.csv", rows)
    (output / "VGUP参数压缩计算说明.md").write_text(
        "# VGUP参数压缩计算说明\n\n"
        f"- ERUP模块参数量：{erup_params:,}\n"
        f"- VGUP模块参数量：{vgup_params:,}\n"
        f"- VGUP/ERUP：{ratio:.8f}\n"
        f"- 1 - VGUP/ERUP：{reduction:.8f}（{reduction * 100:.4f}%）\n\n"
        "以上来自当前真实代码重新实例化统计，不使用“约1/8”等未经核算的描述。"
        "ERUP在当前账号下没有可确认的正式训练指标，因此这里只比较结构复杂度。\n",
        encoding="utf-8",
    )
    metrics = _write_pair_metric_table(
        output / "VGUP加入前后指标对比.csv",
        metric_rows,
        no_vgup,
        with_vgup,
    )
    changes = {row["指标"]: row["变化量"] for row in metrics}
    directions = {
        metric: ("上升" if value > 0 else "下降" if value < 0 else "不变")
        for metric, value in changes.items()
    }
    (output / "VGUP加入前后结果分析.md").write_text(
        "# VGUP加入前后结果分析\n\n"
        f"文件证据显示：P{directions['P']}（{changes['P']:+.6f}），"
        f"R{directions['R']}（{changes['R']:+.6f}），"
        f"mAP50{directions['mAP50']}（{changes['mAP50']:+.6f}），"
        f"mAP50-95{directions['mAP50-95']}（{changes['mAP50-95']:+.6f}）。\n\n"
        "因此不得写成“所有指标均提升”。门控分布与输入统计关系需由GPU Notebook真实运行后解释。\n",
        encoding="utf-8",
    )


def _extract_dpls_cpu(output: Path, metric_rows: list[dict[str, Any]]) -> None:
    _write_pair_metric_table(
        output / "DPLS总体指标对比.csv",
        metric_rows,
        "YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP",
        "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP",
    )


def _extract_shallow_cpu(
    output: Path,
    metric_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
) -> None:
    names = [
        "YOLO11n",
        "YOLO11n + InceptionDW",
        "YOLO11n + PConv",
        "YOLO11n + LSKConv",
    ]
    metrics = [row for row in metric_rows if row["模型"] in names]
    lookup = {row["模型"]: row for row in metrics}
    baseline = lookup["YOLO11n"]
    changes = []
    for row in metrics:
        changes.append(
            {
                "模型": row["模型"],
                "ΔP": float(row["P"]) - float(baseline["P"]),
                "ΔR": float(row["R"]) - float(baseline["R"]),
                "ΔmAP50": float(row["mAP50"]) - float(baseline["mAP50"]),
                "ΔmAP50-95": float(row["mAP50-95"]) - float(baseline["mAP50-95"]),
                "身份状态": row["身份状态"],
            }
        )
    _write_csv(output / "浅层卷积算子指标对比.csv", metrics)
    _write_csv(output / "浅层卷积算子变化量.csv", changes)
    filtered_complexity = [row for row in complexity_rows if row["模型"] in names]
    _write_csv(output / "浅层卷积算子复杂度对比.csv", filtered_complexity)
    best = max(metrics, key=lambda row: float(row["mAP50-95"]))
    caveat = (
        "其中InceptionDW的本地best.pt/results.csv缺失，其指标仅来自任务文档，待人工确认。"
        if lookup["YOLO11n + InceptionDW"]["身份状态"] != "已确认"
        else "所有四个指标行均有本地文件证据。"
    )
    (output / "浅层卷积实验结论.md").write_text(
        "# 浅层卷积实验结论\n\n"
        f"本次统一记录中，{best['模型']}取得最高mAP50-95（{float(best['mAP50-95']):.6f}）。"
        "该结论只描述本次单随机种子实验，不扩展为‘稳定优于其他算子’。\n\n"
        + caveat
        + "\n",
        encoding="utf-8",
    )


def _write_general_guidance(output: Path) -> None:
    (output / "正文实验材料建议.md").write_text(
        "# 正文实验材料建议\n\n"
        "## 高优先级\n\n"
        "1. 总体消融结果与近邻对照差值表。\n"
        "2. DPLS船舶短边分布、P2–P5空间稀释率、PLS/DPLS总体与短边分组结果。\n"
        "3. CA-SCAM与SCAM近邻结果、三层beta及一组真实内部校准响应。\n"
        "4. VGUP与ERUP参数量、VGUP加入前后结果、门控分布与一组输入处理路径。\n"
        "5. 统一复杂度表。\n\n"
        "## 补充材料\n\n"
        "Precision/Recall与loss曲线、完整代表案例候选、失败案例、全部门控统计原始CSV。\n\n"
        "## 尚待GPU结果\n\n"
        "短边条件AP、真实特征响应、CA-SCAM内部图、VGUP门控统计和同图检测对比。\n",
        encoding="utf-8",
    )
    (output / "后续实验接入接口.md").write_text(
        "# 后续实验接入接口\n\n"
        "以下实验本轮不运行，只预留统一字段：模型、数据集、输入尺寸、P、R、mAP50、mAP50-95、AP75、Params、GFLOPs、权重SHA256、配置SHA256、来源与备注。\n\n"
        "- 第二数据集泛化：追加dataset字段并保持同一权重/阈值审计。\n"
        "- YOLOv8n迁移：追加backbone_family字段。\n"
        "- YOLO11s比较：追加scale字段。\n"
        "- SOTA比较：必须记录官方代码/权重/训练协议来源。\n"
        "- Complexity Trade-off：从统一复杂度CSV和正式指标表联接，禁止手抄参数量。\n",
        encoding="utf-8",
    )


def _planned_gpu_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    sha_by_model = {record["identity"]: record["checkpoint_sha256"] for record in audit["records"]}
    plans = [
        ("DPLS", "DPLS短边分组检测结果", "DPLS_短边分组检测结果.csv", ["YOLO11n", "YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP", "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"], "DPLS_实验材料提取_Colab.ipynb", "正文", "表X", "尺度专项评价"),
        ("DPLS", "DPLS特征响应", "DPLS_P2特征响应_图像XXX.png", ["YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP", "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"], "DPLS_实验材料提取_Colab.ipynb", "正文", "图X", "尺度特征分析"),
        ("DPLS", "DPLS代表案例", "DPLS_代表案例候选.csv", ["YOLO11n + InceptionDW + PLS + CA-SCAM + VGUP", "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"], "DPLS_实验材料提取_Colab.ipynb", "补充", "表S", "代表案例"),
        ("CA-SCAM", "CA-SCAM内部机制", "CA-SCAM_内部机制统计.csv", ["YOLO11n + InceptionDW + DPLS + SCAM + VGUP", "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"], "CA-SCAM_实验材料提取_Colab.ipynb", "正文", "图X", "注意力机制分析"),
        ("CA-SCAM", "CA-SCAM代表案例", "CA-SCAM_代表案例候选.csv", ["YOLO11n + InceptionDW + DPLS + SCAM + VGUP", "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"], "CA-SCAM_实验材料提取_Colab.ipynb", "补充", "表S", "代表案例"),
        ("VGUP", "VGUP全验证集门控", "VGUP_全验证集门控统计.csv", ["YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"], "VGUP_实验材料提取_Colab.ipynb", "正文", "图X", "门控机制分析"),
        ("VGUP", "VGUP代表案例", "VGUP_代表案例候选.csv", ["YOLO11n + InceptionDW + DPLS + CA-SCAM", "YOLO11n + InceptionDW + DPLS + CA-SCAM + VGUP"], "VGUP_实验材料提取_Colab.ipynb", "补充", "表S", "代表案例"),
    ]
    rows = []
    for innovation, material, filename, models, notebook, placement, figure, section in plans:
        related_sha = ";".join(sha_by_model[name] for name in models if name in sha_by_model)
        rows.append(
            {
                "创新点": innovation,
                "材料名称": material,
                "文件名": filename,
                "数据来源": notebook,
                "使用模型": "；".join(models),
                "使用权重SHA256": related_sha,
                "是否需要GPU": "是",
                "是否已完成": "否",
                "适合正文/补充材料": placement,
                "建议图号/表号": figure,
                "对应论文小节": section,
                "备注": "运行对应Colab Notebook后生成；禁止手工伪造",
            }
        )
    return rows


def build_material_manifest(material_root: Path, audit: dict[str, Any]) -> Path:
    rows = []
    for category_key, directory_name in CATEGORY_DIRS.items():
        directory = material_root / directory_name
        for path in sorted(directory.iterdir()):
            if (
                not path.is_file()
                or path.name in {"论文实验材料索引.xlsx", "论文实验材料索引_源数据.csv"}
                or path.name.endswith(".inspect.ndjson")
            ):
                continue
            placement = "正文" if any(token in path.name for token in ("指标", "复杂度", "分布", "稀释率", "加入前后")) else "补充"
            rows.append(
                {
                    "创新点": category_key,
                    "材料名称": path.stem,
                    "文件名": path.name,
                    "数据来源": "本地正式best.pt/results.csv、当前代码或已确认尺度分析材料",
                    "使用模型": "见文件内容/权重身份审计",
                    "使用权重SHA256": "见权重身份审计.csv",
                    "是否需要GPU": "否",
                    "是否已完成": "是",
                    "适合正文/补充材料": placement,
                    "建议图号/表号": "待排版",
                    "对应论文小节": category_key,
                    "备注": f"文件SHA256={sha256_file(path)}",
                }
            )
    rows.extend(_planned_gpu_rows(audit))
    destination = material_root / "00_总体材料" / "论文实验材料索引_源数据.csv"
    _write_csv(destination, rows)
    return destination


def build_cpu_materials(
    *,
    results_dir: str | Path,
    material_root: str | Path,
    scale_materials: str | Path,
    repo_root: str | Path = ROOT,
) -> dict[str, Any]:
    repo = Path(repo_root)
    root = Path(material_root)
    directories = _prepare_directories(root)
    print("[1/8] 审计checkpoint、训练日志和模型结构", flush=True)
    audit = audit_experiment_results(results_dir, repo, directories["总体"])
    if audit["errors"]:
        raise RuntimeError(f"Weight audit failures: {audit['errors']}")
    _write_environment_audit(directories["总体"], repo, audit)

    print("[2/8] 生成指标、差值和训练曲线", flush=True)
    supplemental = _supplemental_metrics_file(directories["总体"])
    training = build_training_materials(
        directories["总体"] / "权重身份审计.json",
        directories["总体"],
        supplemental,
    )
    _write_document_metric_reconciliation(directories["总体"], training["metric_rows"])

    print("[3/8] 重新计算所有可构建模型复杂度", flush=True)
    extra_models = [
        ("YOLO11n + InceptionDW", repo / "experiments/formal_ablation_v1/A1_inceptiondw.yaml"),
        ("YOLO11n + PLS + CA-SCAM + ERUP（仅配置）", repo / "experiments/pls_scam_family/PLS_CA_SCAM_ERUP_yolo11n.yaml"),
    ]
    complexity = compute_complexities(
        directories["总体"] / "权重身份审计.json",
        directories["总体"] / "02_所有已有模型复杂度.csv",
        directories["总体"] / "模型图摘要.md",
        imgsz=640,
        extra_models=extra_models,
    )

    print("[4/8] 整理已确认DPLS尺度材料", flush=True)
    _copy_scale_materials(Path(scale_materials), directories["DPLS"])
    _extract_dpls_cpu(directories["DPLS"], training["metric_rows"])

    print("[5/8] 提取CA-SCAM静态参数和近邻结果", flush=True)
    _extract_ca_scam_cpu(audit, directories["CA-SCAM"], training["metric_rows"])

    print("[6/8] 重新计算VGUP/ERUP复杂度和指标变化", flush=True)
    _extract_vgup_cpu(audit, directories["VGUP"], training["metric_rows"], complexity)

    print("[7/8] 整理浅层卷积算子探索", flush=True)
    _extract_shallow_cpu(directories["浅层卷积"], training["metric_rows"], complexity)
    _write_general_guidance(directories["总体"])

    print("[8/8] 生成材料索引源数据并检查目录平坦性", flush=True)
    manifest_path = directories["总体"] / "论文实验材料索引_源数据.csv"
    report = {
        "material_root": str(root.resolve()),
        "directories": {key: str(path.resolve()) for key, path in directories.items()},
        "weights_identified": len(audit["records"]),
        "identity_status_counts": pd.Series(
            [record["identity_status"] for record in audit["records"]]
        ).value_counts().to_dict(),
        "complexity_models": len(complexity),
        "manifest_csv": str(manifest_path.resolve()),
    }
    (directories["总体"] / "CPU材料生成报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Write the report before indexing so a clean first run and every rerun
    # produce the same complete set of manifest rows.
    manifest = build_material_manifest(root, audit)
    nested = [
        str(path)
        for directory in directories.values()
        for path in directory.iterdir()
        if path.is_dir()
    ]
    if nested:
        raise AssertionError(f"Nested material directories are forbidden: {nested}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--material-root", required=True)
    parser.add_argument("--scale-materials", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args()
    report = build_cpu_materials(
        results_dir=args.results_dir,
        material_root=args.material_root,
        scale_materials=args.scale_materials,
        repo_root=args.repo_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
