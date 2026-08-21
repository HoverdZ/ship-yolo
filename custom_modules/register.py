"""Registration helpers for repository-owned modules in Ultralytics.

The helpers provide an idempotent local/Colab registration path without
editing the installed ``site-packages/ultralytics`` source tree.
"""

from __future__ import annotations

import inspect
from types import ModuleType


_PATCH_VERSION = 14


def _set_module_attrs(module: ModuleType, names: dict[str, type]) -> None:
    for name, value in names.items():
        setattr(module, name, value)


def _patch_parse_model(tasks: ModuleType, names: dict[str, type]) -> None:
    """Add repository modules to parse_model's local module sets."""

    parse_model = getattr(tasks, "parse_model", None)
    if parse_model is None:
        raise RuntimeError("ultralytics.nn.tasks.parse_model was not found.")
    if getattr(parse_model, "_ship_yolo_patch_version", 0) == _PATCH_VERSION:
        return

    source = inspect.getsource(parse_model)
    has_c3k2_inception = "C3k2_InceptionDW" in source
    has_c2f_inception = "C2f_InceptionDW" in source
    has_conv_screening = all(
        marker in source
        for marker in ("C3k2_PConv", "C3k2_LSKConv")
    )
    has_cumulative = (
        "elif m is DySample:" in source
        and (
            "elif m is SCAM:" in source
            or "elif m in {SCAM, CASCAM}:" in source
        )
    )
    scam_module_set = "{SCAM, CASCAM}"
    has_calibrated_scam = (
        f"elif m in {scam_module_set}:" in source
    )
    has_adaptive = (
        "elif m in {ERUPPreprocessor, VGUPPreprocessor}:" in source
    )
    comparison_module_set = "{ShuffleAttention, DATBlock}"
    has_comparison_modules = (
        "C2fRepGhost" in source
        and "C2fRFA" in source
        and "SimSPPF" in source
        and f"elif m in {comparison_module_set}:" in source
        and "elif m is FASFF:" in source
        and "elif m is WeightedFeatureFusion:" in source
    )
    has_ac_yolo = "C2PSA_ACmix" in source
    has_dcd_head = "DCDDetect" in source
    if (
        has_c3k2_inception
        and has_c2f_inception
        and has_conv_screening
        and has_cumulative
        and has_adaptive
        and has_calibrated_scam
        and has_comparison_modules
        and has_ac_yolo
        and has_dcd_head
    ):
        parse_model._ship_yolo_patched = True
        parse_model._ship_yolo_patch_version = _PATCH_VERSION
        parse_model._inceptiondw_patched = True
        parse_model._conv_screening_patched = True
        parse_model._cumulative_models_patched = True
        parse_model._adaptive_preprocessors_patched = True
        parse_model._calibrated_scam_patched = True
        parse_model._comparison_modules_patched = True
        parse_model._ac_yolo_patched = True
        parse_model._dcd_head_patched = True
        return

    base_marker = "base_modules = frozenset(\n        {"
    if base_marker not in source:
        raise RuntimeError("Unable to locate parse_model base_modules block for custom registration.")
    base_additions: list[str] = []
    repeat_additions: list[str] = []
    if not has_c3k2_inception:
        base_additions.append("C3k2_InceptionDW")
        repeat_additions.append("C3k2_InceptionDW")
    if not has_c2f_inception:
        base_additions.append("C2f_InceptionDW")
        repeat_additions.append("C2f_InceptionDW")
    if not has_conv_screening:
        base_additions.extend(("C3k2_PConv", "C3k2_LSKConv"))
        repeat_additions.extend(("C3k2_PConv", "C3k2_LSKConv"))
    if not has_comparison_modules:
        base_additions.extend(("C2fRepGhost", "C2fRFA", "SimSPPF"))
        repeat_additions.extend(("C2fRepGhost", "C2fRFA"))
    if not has_ac_yolo:
        base_additions.append("C2PSA_ACmix")
        repeat_additions.append("C2PSA_ACmix")
    if base_additions:
        inserted = "".join(f"            {name},\n" for name in base_additions)
        source = source.replace(
            base_marker,
            f"{base_marker}\n{inserted.rstrip()}",
            1,
        )
    if repeat_additions:
        repeat_marker = "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {"
        if repeat_marker not in source:
            raise RuntimeError("Unable to locate parse_model repeat_modules block.")
        inserted = "".join(f"            {name},\n" for name in repeat_additions)
        source = source.replace(
            repeat_marker,
            f"{repeat_marker}\n{inserted.rstrip()}",
            1,
        )

    if not has_cumulative:
        branch_marker = "        elif m is AIFI:"
        if branch_marker not in source:
            raise RuntimeError(
                "Unable to locate parse_model AIFI branch for DySample/SCAM registration."
            )
        cumulative_branch = """        elif m is DySample:
            if isinstance(f, (list, tuple)):
                raise ValueError("DySample expects exactly one input feature.")
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m in {SCAM, CASCAM}:
            if isinstance(f, (list, tuple)):
                raise ValueError(f"{m.__name__} expects exactly one input feature.")
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
"""
        source = source.replace(
            branch_marker,
            cumulative_branch + branch_marker,
            1,
        )
    elif not has_calibrated_scam:
        if "        elif m in {SCAM, CASCAM}:" in source:
            source = source.replace(
                "        elif m in {SCAM, CASCAM}:",
                "        elif m in {SCAM, CASCAM}:",
                1,
            )
        else:
            source = source.replace(
                "        elif m is SCAM:",
                "        elif m in {SCAM, CASCAM}:",
                1,
            )

    if not has_adaptive:
        branch_marker = "        elif m is AIFI:"
        if branch_marker not in source:
            raise RuntimeError(
                "Unable to locate parse_model AIFI branch for adaptive preprocessors."
            )
        adaptive_branch = """        elif m in {ERUPPreprocessor, VGUPPreprocessor}:
            if isinstance(f, (list, tuple)):
                raise ValueError(f"{m.__name__} expects exactly one RGB input.")
            c1 = ch[f]
            if c1 != 3:
                raise ValueError(
                    f"{m.__name__} must be the first RGB layer, got {c1} channels."
                )
            c2 = c1
            args = [c1, *args]
"""
        source = source.replace(
            branch_marker,
            adaptive_branch + branch_marker,
            1,
        )

    if not has_comparison_modules:
        branch_marker = "        elif m is AIFI:"
        if branch_marker not in source:
            raise RuntimeError(
                "Unable to locate parse_model AIFI branch for comparison modules."
            )
        comparison_branch = """        elif m in {ShuffleAttention, DATBlock}:
            if isinstance(f, (list, tuple)):
                raise ValueError(f"{m.__name__} expects exactly one feature tensor.")
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m is FASFF:
            if not isinstance(f, (list, tuple)) or len(f) != 4:
                raise ValueError("FASFF requires four PAN-4 features.")
            input_channels = [ch[index] for index in f]
            target_index = int(args[0])
            c2 = input_channels[target_index]
            args = [input_channels, target_index]
        elif m is WeightedFeatureFusion:
            if not isinstance(f, (list, tuple)):
                raise ValueError("WeightedFeatureFusion requires a feature list.")
            input_channels = [ch[index] for index in f]
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [input_channels, c2, *args[1:]]
"""
        source = source.replace(branch_marker, comparison_branch + branch_marker, 1)

    if not has_dcd_head:
        detect_marker = "                Detect,\n                WorldDetect,"
        if detect_marker not in source:
            raise RuntimeError("Unable to locate the Detect parser set for DCDDetect registration.")
        source = source.replace(
            detect_marker,
            "                Detect,\n                DCDDetect,\n                WorldDetect,",
            1,
        )
        legacy_marker = "if m in {Detect, YOLOEDetect,"
        if legacy_marker not in source:
            raise RuntimeError("Unable to locate the Detect legacy set for DCDDetect registration.")
        source = source.replace(
            legacy_marker,
            "if m in {Detect, DCDDetect, YOLOEDetect,",
            1,
        )

    namespace = tasks.__dict__
    namespace.update(names)
    exec(
        compile(
            source,
            inspect.getsourcefile(parse_model) or "<ship_yolo_parse_model>",
            "exec",
        ),
        namespace,
    )
    tasks.parse_model._ship_yolo_patched = True
    tasks.parse_model._ship_yolo_patch_version = _PATCH_VERSION
    tasks.parse_model._inceptiondw_patched = True
    tasks.parse_model._conv_screening_patched = True
    tasks.parse_model._cumulative_models_patched = True
    tasks.parse_model._adaptive_preprocessors_patched = True
    tasks.parse_model._calibrated_scam_patched = True
    tasks.parse_model._comparison_modules_patched = True
    tasks.parse_model._ac_yolo_patched = True
    tasks.parse_model._dcd_head_patched = True


def register_custom_modules(patch_parse_model: bool = True) -> None:
    """Register every repository-owned module through one idempotent path."""

    from custom_modules.c2f_inceptiondw import C2f_InceptionDW
    from custom_modules.ac_yolo_official import ACmix, C2PSA_ACmix
    from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
    from custom_modules.c3k2_conv_screening import (
        C3k2_LSKConv,
        C3k2_PConv,
    )
    from custom_modules.calibrated_scam import CASCAM
    from custom_modules.dysample import DySample
    from custom_modules.dcd_head import DCDDetect
    from custom_modules.erup import ERUPPreprocessor
    from custom_modules.scam import SCAM
    from custom_modules.vgup import VGUPPreprocessor
    from custom_modules.remote_ship_reproductions import (
        C2fRFA,
        C2fRepGhost,
        DATBlock,
        FASFF,
        ShuffleAttention,
        SimSPPF,
        WeightedFeatureFusion,
    )
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    names = {
        "ACmix": ACmix,
        "C2f_InceptionDW": C2f_InceptionDW,
        "C2PSA_ACmix": C2PSA_ACmix,
        "C3k2_InceptionDW": C3k2_InceptionDW,
        "C3k2_LSKConv": C3k2_LSKConv,
        "C3k2_PConv": C3k2_PConv,
        "CASCAM": CASCAM,
        "DySample": DySample,
        "DCDDetect": DCDDetect,
        "ERUPPreprocessor": ERUPPreprocessor,
        "SCAM": SCAM,
        "VGUPPreprocessor": VGUPPreprocessor,
        "C2fRFA": C2fRFA,
        "C2fRepGhost": C2fRepGhost,
        "DATBlock": DATBlock,
        "FASFF": FASFF,
        "ShuffleAttention": ShuffleAttention,
        "SimSPPF": SimSPPF,
        "WeightedFeatureFusion": WeightedFeatureFusion,
    }
    _set_module_attrs(modules, names)
    _set_module_attrs(tasks, names)

    if patch_parse_model:
        _patch_parse_model(tasks, names)


def register_inceptiondw_modules(patch_parse_model: bool = True) -> None:
    """Register the InceptionDW C3k2 module and shared repository modules."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_conv_screening_modules(patch_parse_model: bool = True) -> None:
    """Register the paper-reported controlled PConv/LSKConv variants."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_cumulative_modules(patch_parse_model: bool = True) -> None:
    """Register DySample, SCAM, InceptionDW, and shared repository modules."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_adaptive_preprocessors(patch_parse_model: bool = True) -> None:
    """Register VGUP and its shared image-processing primitives."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_calibrated_scam_modules(
    patch_parse_model: bool = True,
) -> None:
    """Register CA-SCAM and every shared repository module."""

    register_custom_modules(patch_parse_model=patch_parse_model)
