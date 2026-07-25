"""Registration helpers for repository-owned modules in Ultralytics.

The helpers provide an idempotent local/Colab registration path without
editing the installed ``site-packages/ultralytics`` source tree.
"""

from __future__ import annotations

import inspect
from types import ModuleType


_PATCH_VERSION = 4


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
    has_ablation = all(
        marker in source
        for marker in ("C3k2CrossConv", "CGFM", "AlignConcat", "DD")
    )
    has_inception = "C3k2_InceptionDW" in source
    has_sa_dwpn = "elif m is SDWF:" in source and "DWDown" in source
    has_cumulative = (
        "elif m is DySample:" in source
        and "elif m is SCAM:" in source
    )
    if has_ablation and has_inception and has_sa_dwpn and has_cumulative:
        parse_model._ship_yolo_patched = True
        parse_model._ship_yolo_patch_version = _PATCH_VERSION
        parse_model._sa_dwpn_patched = True
        parse_model._inceptiondw_patched = True
        parse_model._module_ablation_patched = True
        parse_model._cumulative_models_patched = True
        return

    base_marker = "base_modules = frozenset(\n        {"
    if base_marker not in source:
        raise RuntimeError("Unable to locate parse_model base_modules block for custom registration.")
    base_additions: list[str] = []
    repeat_additions: list[str] = []
    if not has_sa_dwpn:
        base_additions.extend(("Align", "DWDown"))
    if not has_ablation:
        base_additions.extend(("C3k2CrossConv", "DD"))
        repeat_additions.append("C3k2CrossConv")
    if not has_inception:
        base_additions.append("C3k2_InceptionDW")
        repeat_additions.append("C3k2_InceptionDW")
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

    if not has_ablation:
        c3k2_marker = "            if m is C3k2:  # for M/L/X sizes"
        if c3k2_marker not in source:
            raise RuntimeError("Unable to locate parse_model C3k2 scale branch.")
        source = source.replace(
            c3k2_marker,
            "            if m in {C3k2, C3k2CrossConv}:  # for M/L/X sizes",
            1,
        )

        branch_marker = "        elif m is AIFI:"
        if branch_marker not in source:
            raise RuntimeError("Unable to locate parse_model AIFI branch for fusion registration.")
        fusion_branch = """        elif m in {CGFM, AlignConcat}:
            if not isinstance(f, (list, tuple)) or len(f) != 2:
                raise ValueError(f"{m.__name__} requires [deep_upsampled, shallow_lateral].")
            c_deep, c_shallow = ch[f[0]], ch[f[1]]
            c2 = 2 * c_shallow
            args = [c_deep, c_shallow, *args]
"""
        source = source.replace(branch_marker, fusion_branch + branch_marker, 1)

    if not has_sa_dwpn:
        branch_marker = "        elif m is AIFI:"
        if branch_marker not in source:
            raise RuntimeError("Unable to locate parse_model AIFI branch for custom registration.")
        sdwf_branch = """        elif m is SDWF:
            if not isinstance(f, (list, tuple)):
                f = [f]
            c1, c2 = ch[f[0]], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
"""
        source = source.replace(branch_marker, sdwf_branch + branch_marker, 1)

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
        elif m is SCAM:
            if isinstance(f, (list, tuple)):
                raise ValueError("SCAM expects exactly one input feature.")
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
"""
        source = source.replace(
            branch_marker,
            cumulative_branch + branch_marker,
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
    tasks.parse_model._sa_dwpn_patched = True
    tasks.parse_model._inceptiondw_patched = True
    tasks.parse_model._module_ablation_patched = True
    tasks.parse_model._cumulative_models_patched = True


def register_custom_modules(patch_parse_model: bool = True) -> None:
    """Register every repository-owned module through one idempotent path."""

    from custom_modules.c3k2_crossconv import C3k2CrossConv
    from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
    from custom_modules.cgfm import AlignConcat, CGFM
    from custom_modules.dd import DD
    from custom_modules.dysample import DySample
    from custom_modules.sa_dwpn import Align, DWDown, SDWF
    from custom_modules.scam import SCAM
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    names = {
        "Align": Align,
        "AlignConcat": AlignConcat,
        "C3k2CrossConv": C3k2CrossConv,
        "C3k2_InceptionDW": C3k2_InceptionDW,
        "CGFM": CGFM,
        "DD": DD,
        "DySample": DySample,
        "DWDown": DWDown,
        "SCAM": SCAM,
        "SDWF": SDWF,
    }
    _set_module_attrs(modules, names)
    _set_module_attrs(tasks, names)

    if patch_parse_model:
        _patch_parse_model(tasks, names)


def register_sa_dwpn_modules(patch_parse_model: bool = True) -> None:
    """Backward-compatible SA-DWPN registration entrypoint."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_inceptiondw_modules(patch_parse_model: bool = True) -> None:
    """Register the InceptionDW C3k2 module and shared repository modules."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_module_ablation_modules(patch_parse_model: bool = True) -> None:
    """Register CrossConv, DD, CGFM, and shared repository modules."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_cumulative_modules(patch_parse_model: bool = True) -> None:
    """Register DySample, SCAM, InceptionDW, and shared repository modules."""

    register_custom_modules(patch_parse_model=patch_parse_model)
