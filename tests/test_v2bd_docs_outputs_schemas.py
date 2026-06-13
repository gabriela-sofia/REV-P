"""v2bd docs, outputs and schema tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_eight_schemas_and_summary_exist():
    assert len(list((ROOT / "docs").glob("v2bd_*.md"))) == 3
    schemas = list((ROOT / "datasets" / "schemas").glob("v2bd_*.schema.json"))
    assert len(schemas) == 8
    assert all(json.loads(path.read_text(encoding="utf-8"))["required"] for path in schemas)
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2bd_sentinel_patch_footprint_recovery_summary.json").read_text(encoding="utf-8"))
    assert summary["boundary_recovered"] is True
    assert summary["ready_patch_feed_rows"] == 1
    assert summary["turning_point_level"] == "TP1_ONE_PATCH_BOUNDARY_CANDIDATE_REQUIRES_HUMAN_REVIEW"
