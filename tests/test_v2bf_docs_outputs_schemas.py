"""v2bf artifact completeness tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_outputs_and_schemas_exist():
    assert len(list((ROOT / "docs").glob("v2bf_*.md"))) == 4
    assert len(list((ROOT / "datasets/schemas").glob("v2bf_*.schema.json"))) == 13
    for path in (ROOT / "datasets/schemas").glob("v2bf_*.schema.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["type"] == "object"
    assert (ROOT / "outputs_public/execution_reports/v2bf_recife_observed_event_polygon_tp2_summary.json").is_file()
