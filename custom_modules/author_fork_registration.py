"""Registration adapters for the original-author YOLOv12/YOLOv13 forks.

These forks both report Ultralytics 8.3.63 but carry architecture-specific
model parsers. The helpers extend the active author's parser in memory; they
do not edit the cloned author repository or installed site-packages.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from types import ModuleType


_AUTHOR_PATCH_VERSION = 2
_AUTHOR_SOURCES = {
    "yolov12": (
        "https://github.com/sunsmarterjie/yolov12",
        "01a22c0603e0eaa6d9bd62120a391e744d92cea2",
    ),
    "yolov13": (
        "https://github.com/iMoonLab/yolov13",
        "70f23ede45ee00a30cf6139c3d1ea7abe3df4eec",
    ),
}


def _set_module_attrs(module: ModuleType, names: dict[str, type]) -> None:
    for name, value in names.items():
        setattr(module, name, value)


def _replace_once(source: str, marker: str, replacement: str, label: str) -> str:
    if source.count(marker) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} marker in the fixed author parser; "
            f"found {source.count(marker)}."
        )
    return source.replace(marker, replacement, 1)


def _replace_first(source: str, marker: str, replacement: str, label: str) -> str:
    if marker not in source:
        raise RuntimeError(f"Unable to locate the {label} marker in the fixed author parser.")
    return source.replace(marker, replacement, 1)


def _validate_author_fork(tasks: ModuleType, fork: str) -> None:
    import ultralytics

    if ultralytics.__version__ != "8.3.63":
        raise RuntimeError(
            f"{fork} registration requires its author fork reporting "
            f"Ultralytics 8.3.63, got {ultralytics.__version__}."
        )
    has_hyperace = all(
        hasattr(tasks, name)
        for name in ("HyperACE", "DownsampleConv", "FullPAD_Tunnel")
    )
    if fork == "yolov12" and has_hyperace:
        raise RuntimeError("The active parser is YOLOv13, not the YOLOv12 author fork.")
    if fork == "yolov13" and not has_hyperace:
        raise RuntimeError("The active parser is not the YOLOv13 author fork.")
    if not hasattr(tasks, "A2C2f"):
        raise RuntimeError(f"The active {fork} parser does not expose A2C2f.")

    package_path = Path(ultralytics.__file__).resolve()
    repository = next(
        (parent for parent in package_path.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        raise RuntimeError(
            f"The active {fork} package is not imported from an author Git clone: "
            f"{package_path}."
        )

    remote = subprocess.check_output(
        ["git", "-C", str(repository), "remote", "get-url", "origin"],
        text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    expected_remote, expected_commit = _AUTHOR_SOURCES[fork]
    normalized_remote = remote.removesuffix(".git").replace(
        "git@github.com:",
        "https://github.com/",
    )
    if normalized_remote.casefold() != expected_remote.casefold():
        raise RuntimeError(
            f"Expected {fork} author remote {expected_remote}, got {remote}."
        )
    if commit != expected_commit:
        raise RuntimeError(
            f"Expected {fork} author commit {expected_commit}, got {commit}."
        )


def _patch_author_parse_model(
    tasks: ModuleType,
    names: dict[str, type],
    fork: str,
) -> None:
    parse_model = getattr(tasks, "parse_model", None)
    if parse_model is None:
        raise RuntimeError("ultralytics.nn.tasks.parse_model was not found.")
    if (
        getattr(parse_model, "_ship_yolo_author_patch_version", 0)
        == _AUTHOR_PATCH_VERSION
        and getattr(parse_model, "_ship_yolo_author_fork", None) == fork
    ):
        return

    source = inspect.getsource(parse_model).replace("\r\n", "\n")
    if "CGDR" in source or "KBLLitePreprocessor" in source:
        raise RuntimeError(
            "The active author parser already contains an unknown custom patch; "
            "start from a fresh Runtime."
        )

    base_marker = "            A2C2f,\n"
    source = _replace_first(
        source,
        base_marker,
        base_marker + "            CGDR,\n",
        "A2C2f base-module",
    )

    branch_marker = "        elif m is AIFI:\n"
    custom_branches = """        elif m in {KBLLitePreprocessor, VGUPPreprocessor}:
            if isinstance(f, (list, tuple)):
                raise ValueError(f"{m.__name__} expects exactly one RGB input.")
            c1 = ch[f]
            if c1 != 3:
                raise ValueError(
                    f"{m.__name__} must be the first RGB layer, got {c1} channels."
                )
            c2 = c1
            args = [c1, *args]
        elif m is DySample:
            if isinstance(f, (list, tuple)):
                raise ValueError("DySample expects exactly one feature tensor.")
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
"""
    source = _replace_once(
        source,
        branch_marker,
        custom_branches + branch_marker,
        "AIFI branch",
    )

    namespace = tasks.__dict__
    namespace.update(names)
    exec(
        compile(
            source,
            inspect.getsourcefile(parse_model) or f"<{fork}_author_parse_model>",
            "exec",
        ),
        namespace,
    )
    tasks.parse_model._ship_yolo_author_patch_version = _AUTHOR_PATCH_VERSION
    tasks.parse_model._ship_yolo_author_fork = fork


def _register_author_modules(fork: str) -> None:
    from custom_modules.cgdr import CGDR
    from custom_modules.dysample import DySample
    from custom_modules.kbl_lite import KBLLitePreprocessor
    from custom_modules.vgup import VGUPPreprocessor
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    _validate_author_fork(tasks, fork)
    names = {
        "CGDR": CGDR,
        "DySample": DySample,
        "KBLLitePreprocessor": KBLLitePreprocessor,
        "VGUPPreprocessor": VGUPPreprocessor,
    }
    _set_module_attrs(modules, names)
    _set_module_attrs(tasks, names)
    _patch_author_parse_model(tasks, names, fork)


def register_yolov12_author_modules() -> None:
    """Register final ship modules into sunsmarterjie/yolov12@01a22c0."""

    _register_author_modules("yolov12")


def register_yolov13_author_modules() -> None:
    """Register final ship modules into iMoonLab/yolov13@70f23ed."""

    _register_author_modules("yolov13")


__all__ = [
    "register_yolov12_author_modules",
    "register_yolov13_author_modules",
]
