from __future__ import annotations

import json

from tools.generate_incdw_ca_scam_notebooks import generate
from tools.incdw_ca_scam_experiments import RUN_IDS
from tools.validate_incdw_ca_scam_notebooks import validate_all


def test_generated_notebooks_are_pinned_direct_and_independent(tmp_path) -> None:
    commit = "3" * 40
    outputs = generate(commit, output_root=tmp_path)
    report = validate_all(root=tmp_path)
    assert report["passed"]
    assert report["fixed_commit"] == commit
    assert len(outputs) == 2
    assert len({path.name for path in outputs}) == 2

    found_run_ids = set()
    for path in outputs:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )
        assert "RUN_TRAINING" not in source
        assert "train_foreground(" in source
        assert "/content/ship_detection/data" in source
        assert "GITHUB_TOKEN" in source
        found_run_ids.update(run_id for run_id in RUN_IDS if run_id in source)
    assert found_run_ids == set(RUN_IDS)
