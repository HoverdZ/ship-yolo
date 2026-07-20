"""Registration helpers for repository-owned modules in Ultralytics.

This helper provides a repeatable local/Colab registration path without editing
the installed ``site-packages/ultralytics`` source tree.
"""

from __future__ import annotations

import inspect
from types import ModuleType


def _set_module_attrs(module: ModuleType, names: dict[str, type]) -> None:
    for name, value in names.items():
        setattr(module, name, value)


def _patch_parse_model(tasks: ModuleType, names: dict[str, type]) -> None:
    parse_model = getattr(tasks, "parse_model", None)
    if parse_model is None:
        raise RuntimeError("ultralytics.nn.tasks.parse_model was not found.")
    if getattr(parse_model, "_ship_yolo_patched", False):
        return

    source = inspect.getsource(parse_model)

    if "SDWF" in source and "DWDown" in source and "C3k2_SCConv" in source:
        parse_model._ship_yolo_patched = True
        parse_model._sa_dwpn_patched = True
        parse_model._scconv_patched = True
        return

    base_marker = "base_modules = frozenset(\n        {"
    if base_marker not in source:
        raise RuntimeError("Unable to locate parse_model base_modules block for custom registration.")
    source = source.replace(
        base_marker,
        "base_modules = frozenset(\n        {\n            Align,\n            C3k2_SCConv,\n            DWDown,",
        1,
    )

    repeat_marker = "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {"
    if repeat_marker not in source:
        raise RuntimeError(
            "Unable to locate parse_model repeat_modules block for SCConv registration."
        )
    source = source.replace(
        repeat_marker,
        "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {\n            C3k2_SCConv,",
        1,
    )

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
    tasks.parse_model._sa_dwpn_patched = True
    tasks.parse_model._scconv_patched = True


def register_custom_modules(patch_parse_model: bool = True) -> None:
    """Register all ship-yolo custom modules with Ultralytics.

    This function is idempotent. Calling it more than once in the same process is
    safe and does not stack multiple wrappers.
    """

    from custom_modules.c3k2_scconv import C3k2_SCConv
    from custom_modules.sa_dwpn import Align, DWDown, SDWF
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    names = {
        "Align": Align,
        "C3k2_SCConv": C3k2_SCConv,
        "DWDown": DWDown,
        "SDWF": SDWF,
    }
    _set_module_attrs(modules, names)
    _set_module_attrs(tasks, names)

    if patch_parse_model:
        _patch_parse_model(tasks, names)


def register_sa_dwpn_modules(patch_parse_model: bool = True) -> None:
    """Backward-compatible registration entrypoint used by SA-DWPN tools."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_scconv_modules(patch_parse_model: bool = True) -> None:
    """Register SCConv modules through the shared repository mechanism."""

    register_custom_modules(patch_parse_model=patch_parse_model)
