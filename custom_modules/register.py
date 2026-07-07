"""Registration helpers for SA-DWPN modules in Ultralytics environments.

The preferred long-term path is to copy ``sa_dwpn.py`` into
``ultralytics/nn/modules`` and add a small permanent ``parse_model`` branch.
This helper provides a repeatable repository-owned fallback for Colab or local
research runs where editing installed Ultralytics files is inconvenient.
"""

from __future__ import annotations

import inspect
from types import ModuleType

from custom_modules.sa_dwpn import Align, DWDown, SDWF


def _set_module_attrs(module: ModuleType, names: dict[str, type]) -> None:
    for name, value in names.items():
        setattr(module, name, value)


def _patch_parse_model(tasks: ModuleType) -> None:
    parse_model = getattr(tasks, "parse_model", None)
    if parse_model is None:
        raise RuntimeError("ultralytics.nn.tasks.parse_model was not found.")
    if getattr(parse_model, "_sa_dwpn_patched", False):
        return

    source = inspect.getsource(parse_model)

    if "SDWF" in source and "DWDown" in source:
        parse_model._sa_dwpn_patched = True
        return

    base_marker = "base_modules = frozenset(\n        {"
    if base_marker not in source:
        raise RuntimeError("Unable to locate parse_model base_modules block for SA-DWPN registration.")
    source = source.replace(
        base_marker,
        "base_modules = frozenset(\n        {\n            Align,\n            DWDown,",
        1,
    )

    branch_marker = "        elif m is AIFI:"
    if branch_marker not in source:
        raise RuntimeError("Unable to locate parse_model AIFI branch for SA-DWPN registration.")
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
    namespace.update({"Align": Align, "DWDown": DWDown, "SDWF": SDWF})
    exec(compile(source, inspect.getsourcefile(parse_model) or "<sa_dwpn_parse_model>", "exec"), namespace)
    tasks.parse_model._sa_dwpn_patched = True


def register_sa_dwpn_modules(patch_parse_model: bool = True) -> None:
    """Register SA-DWPN modules with Ultralytics.

    This function is idempotent. Calling it more than once in the same process is
    safe and does not stack multiple wrappers.
    """

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    names = {"Align": Align, "DWDown": DWDown, "SDWF": SDWF}
    _set_module_attrs(modules, names)
    _set_module_attrs(tasks, names)

    if patch_parse_model:
        _patch_parse_model(tasks)
