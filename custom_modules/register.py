"""Registration helpers for repository-owned modules in Ultralytics.

The helpers provide an idempotent local/Colab registration path without
editing the installed ``site-packages/ultralytics`` source tree.
"""

from __future__ import annotations

import inspect
from types import ModuleType


_PATCH_VERSION = 7


def _set_module_attrs(module: ModuleType, names: dict[str, type]) -> None:
    for name, value in names.items():
        setattr(module, name, value)


def _patch_parse_model(tasks: ModuleType, names: dict[str, type]) -> None:
    """Add repository modules and explicit custom channel inference."""

    parse_model = getattr(tasks, "parse_model", None)
    if parse_model is None:
        raise RuntimeError("ultralytics.nn.tasks.parse_model was not found.")
    if getattr(parse_model, "_ship_yolo_patch_version", 0) == _PATCH_VERSION:
        return

    source = inspect.getsource(parse_model)
    required_names = (
        "ASCGDAlignC3",
        "ASCGDAlignC4",
        "ASCGDAlignC5",
        "ASCGDChannelP4",
        "ASCGDChannelP5",
        "ASCGDDirectP3",
        "ASCGDDirectP4",
        "ASCGDDirectP5",
        "ASCGDGather",
        "ASCGDSpatialP3",
        "ASCGDSwappedP3",
        "ASCGDSwappedP4",
        "ASCGDSwappedP5",
        "ASCGDSymmetricP3",
        "ASCGDSymmetricP4",
        "ASCGDSymmetricP5",
        "C3k2_BADC",
        "C3k2_InceptionDW",
        "FaPNAlign",
        "FaPNAlignmentOnly",
        "FaPNFeatureSelectionKeep",
        "FaPNLateral",
        "FaPNOutputConv",
        "P2GaussianAuxDetect",
        "SCSharedDetect",
        "SemanticConfirmationGate",
        "SDWF",
        "SPDDown",
    )
    if all(name in source for name in required_names):
        parse_model._ship_yolo_patched = True
        parse_model._ship_yolo_patch_version = _PATCH_VERSION
        return

    base_marker = "base_modules = frozenset(\n        {"
    if base_marker not in source:
        raise RuntimeError("Unable to locate parse_model base_modules block for custom registration.")
    source = source.replace(
        base_marker,
        "base_modules = frozenset(\n        {\n            Align,\n            C3k2_BADC,\n"
        "            C3k2_InceptionDW,\n"
        "            DWDown,\n            FaPNLateral,\n            FaPNOutputConv,\n"
        "            SPDDown,",
        1,
    )

    repeat_marker = "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {"
    if repeat_marker not in source:
        raise RuntimeError("Unable to locate parse_model repeat_modules block.")
    source = source.replace(
        repeat_marker,
        "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {\n"
        "            C3k2_BADC,\n            C3k2_InceptionDW,",
        1,
    )

    branch_marker = "        elif m is AIFI:"
    if branch_marker not in source:
        raise RuntimeError("Unable to locate parse_model AIFI branch for custom registration.")
    custom_branches = """        elif m in {ASCGDAlignC3, ASCGDAlignC4, ASCGDAlignC5}:
            if isinstance(f, (list, tuple)):
                raise ValueError(f"{m.__name__} requires exactly one input index.")
            c1, c2 = ch[f], args[0]
            args = [c1, *args]
        elif m is ASCGDGather:
            if not isinstance(f, (list, tuple)) or len(f) != 3:
                raise ValueError("ASCGDGather requires [A3, A4, A5].")
            c2 = args[0]
            args = [[ch[x] for x in f], *args]
        elif m in {ASCGDDirectP3, ASCGDSpatialP3, ASCGDSwappedP3, ASCGDSymmetricP3}:
            if not isinstance(f, (list, tuple)) or len(f) != 2:
                raise ValueError(f"{m.__name__} requires [C3, G].")
            c2 = ch[f[0]]
            args = [[ch[x] for x in f], *args]
        elif m in {ASCGDDirectP4, ASCGDChannelP4, ASCGDSwappedP4, ASCGDSymmetricP4}:
            if not isinstance(f, (list, tuple)) or len(f) != 3:
                raise ValueError(f"{m.__name__} requires [A3, A4, G].")
            c2 = ch[f[1]]
            args = [[ch[x] for x in f], *args]
        elif m in {ASCGDDirectP5, ASCGDChannelP5, ASCGDSwappedP5, ASCGDSymmetricP5}:
            if not isinstance(f, (list, tuple)) or len(f) != 3:
                raise ValueError(f"{m.__name__} requires [P4, C5, G].")
            c2 = ch[f[1]]
            args = [[ch[x] for x in f], *args]
        elif m is FaPNFeatureSelectionKeep:
            if isinstance(f, (list, tuple)):
                raise ValueError("FaPNFeatureSelectionKeep requires exactly one input index.")
            c1 = c2 = ch[f]
            args = [c1, *args]
        elif m is FaPNAlignmentOnly:
            if not isinstance(f, (list, tuple)) or len(f) != 2:
                raise ValueError("FaPNAlignmentOnly requires [selected_low, upsampled_high].")
            c_low, c_high = ch[f[0]], ch[f[1]]
            c2 = c_high
            args = [c_low, c_high, *args]
        elif m is FaPNAlign:
            if not isinstance(f, (list, tuple)) or len(f) != 2:
                raise ValueError("FaPNAlign requires exactly [lateral, top-down] input indices.")
            c_lateral, c_topdown = ch[f[0]], ch[f[1]]
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c_lateral, c_topdown, c2, *args[1:]]
        elif m is SDWF:
            if not isinstance(f, (list, tuple)):
                f = [f]
            c1, c2 = ch[f[0]], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is SemanticConfirmationGate:
            if not isinstance(f, (list, tuple)) or len(f) != 2:
                raise ValueError("SemanticConfirmationGate requires [C3, upsampled P4].")
            c2 = sum(ch[x] for x in f)
            args = [[ch[x] for x in f], *args]
"""
    source = source.replace(branch_marker, custom_branches + branch_marker, 1)

    head_marker = "        elif m is SemanticSegment:"
    if head_marker not in source:
        raise RuntimeError("Unable to locate parse_model SemanticSegment branch.")
    source = source.replace(
        head_marker,
        """        elif m is P2GaussianAuxDetect:
            if not isinstance(f, (list, tuple)) or len(f) != 4:
                raise ValueError("P2GaussianAuxDetect requires [P2, P3, P4, P5].")
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            m.legacy = legacy
        elif m is SCSharedDetect:
            if not isinstance(f, (list, tuple)) or len(f) != 3:
                raise ValueError("SCSharedDetect requires [P3, P4, P5].")
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            m.legacy = legacy
"""
        + head_marker,
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
    tasks.parse_model._fapn_patched = True
    tasks.parse_model._fapn_prefusion_patched = True


def register_custom_modules(patch_parse_model: bool = True) -> None:
    """Register every repository-owned module through one idempotent path."""

    from custom_modules.ascgd import (
        ASCGDAlignC3,
        ASCGDAlignC4,
        ASCGDAlignC5,
        ASCGDChannelP4,
        ASCGDChannelP5,
        ASCGDDirectP3,
        ASCGDDirectP4,
        ASCGDDirectP5,
        ASCGDGather,
        ASCGDSpatialP3,
        ASCGDSwappedP3,
        ASCGDSwappedP4,
        ASCGDSwappedP5,
        ASCGDSymmetricP3,
        ASCGDSymmetricP4,
        ASCGDSymmetricP5,
    )
    from custom_modules.badc import C3k2_BADC
    from custom_modules.c3k2_inceptiondw import C3k2_InceptionDW
    from custom_modules.fapn import FaPNAlign, FaPNLateral, FaPNOutputConv
    from custom_modules.fapn_prefusion import FaPNAlignmentOnly, FaPNFeatureSelectionKeep
    from custom_modules.p2_gaussian_aux import P2GaussianAuxDetect
    from custom_modules.scshared_head import SCSharedDetect
    from custom_modules.scg import SemanticConfirmationGate
    from custom_modules.sa_dwpn import Align, DWDown, SDWF
    from custom_modules.spd import SPDDown
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    names = {
        "ASCGDAlignC3": ASCGDAlignC3,
        "ASCGDAlignC4": ASCGDAlignC4,
        "ASCGDAlignC5": ASCGDAlignC5,
        "ASCGDChannelP4": ASCGDChannelP4,
        "ASCGDChannelP5": ASCGDChannelP5,
        "ASCGDDirectP3": ASCGDDirectP3,
        "ASCGDDirectP4": ASCGDDirectP4,
        "ASCGDDirectP5": ASCGDDirectP5,
        "ASCGDGather": ASCGDGather,
        "ASCGDSpatialP3": ASCGDSpatialP3,
        "ASCGDSwappedP3": ASCGDSwappedP3,
        "ASCGDSwappedP4": ASCGDSwappedP4,
        "ASCGDSwappedP5": ASCGDSwappedP5,
        "ASCGDSymmetricP3": ASCGDSymmetricP3,
        "ASCGDSymmetricP4": ASCGDSymmetricP4,
        "ASCGDSymmetricP5": ASCGDSymmetricP5,
        "Align": Align,
        "C3k2_BADC": C3k2_BADC,
        "C3k2_InceptionDW": C3k2_InceptionDW,
        "DWDown": DWDown,
        "FaPNAlign": FaPNAlign,
        "FaPNAlignmentOnly": FaPNAlignmentOnly,
        "FaPNFeatureSelectionKeep": FaPNFeatureSelectionKeep,
        "FaPNLateral": FaPNLateral,
        "FaPNOutputConv": FaPNOutputConv,
        "P2GaussianAuxDetect": P2GaussianAuxDetect,
        "SCSharedDetect": SCSharedDetect,
        "SemanticConfirmationGate": SemanticConfirmationGate,
        "SDWF": SDWF,
        "SPDDown": SPDDown,
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


def register_fapn_modules(patch_parse_model: bool = True) -> None:
    """Register official FaPN port modules and shared repository modules."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_fapn_prefusion_modules(patch_parse_model: bool = True) -> None:
    """Register FaPN-Prefusion modules and the shared parser patch."""

    register_custom_modules(patch_parse_model=patch_parse_model)


def register_ascgd_modules(patch_parse_model: bool = True) -> None:
    """Register ASCGD modules and the shared parser patch."""

    register_custom_modules(patch_parse_model=patch_parse_model)
