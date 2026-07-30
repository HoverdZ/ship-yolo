"""Reject identity, reuse, selection, and protocol inconsistencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.formal_experiments.registry import load_registry
from tools.paper_artifacts.results.common import _manifest_path


def validate(root: str | Path) -> dict[str, object]:
    base = Path(root)
    registry = load_registry()
    errors: list[str] = []
    completed = 0
    shared_protocol: dict[str, object] | None = None
    protocol_keys = (
        "imgsz",
        "epochs",
        "batch",
        "workers",
        "optimizer",
        "lr0",
        "weight_decay",
    )
    for run_id, run in registry["canonical_runs"].items():
        path = _manifest_path(base, run_id, int(run["seed"]))
        if not path.is_file():
            continue
        completed += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_id") != run_id:
            errors.append(f"{run_id}: manifest identity mismatch")
        if payload.get("test_used_for_selection") is not False:
            errors.append(f"{run_id}: test-selection flag is not false")
        if payload.get("initialization_weight") != run["initialization_weight"]:
            errors.append(f"{run_id}: initialization weight mismatch")
        if payload.get("staged_checkpoint_used") is not False:
            errors.append(f"{run_id}: staged checkpoint was used")
        current = {
            key: payload.get("training", {}).get(key)
            for key in protocol_keys
        }
        if run["dataset_id"] == registry["primary_dataset_id"]:
            if shared_protocol is None:
                shared_protocol = current
            elif current != shared_protocol:
                errors.append(f"{run_id}: formal training protocol drift")
    report = {
        "completed_manifests": completed,
        "registered_runs": len(registry["canonical_runs"]),
        "errors": errors,
        "passed": not errors,
    }
    if errors:
        raise ValueError("\n".join(errors))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    args = parser.parse_args()
    print(json.dumps(validate(args.run_root), indent=2))


if __name__ == "__main__":
    main()
