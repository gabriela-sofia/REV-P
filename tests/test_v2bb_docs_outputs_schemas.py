"""v2bb documentation, output and schema tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_outputs_and_eight_schemas_exist():
    assert len(list((ROOT/"docs").glob("v2bb_*.md"))) == 3
    schemas = list((ROOT/"datasets"/"schemas").glob("v2bb_*.schema.json"))
    assert len(schemas) == 8
    for path in schemas:
        assert json.loads(path.read_text(encoding="utf-8"))["required"]
    summary = json.loads((ROOT/"outputs_public"/"execution_reports"/"v2bb_public_geometry_retrieval_feed_builder_summary.json").read_text(encoding="utf-8"))
    assert summary["turning_point_level"] == "TP0_DOCUMENTED_ABSENCE_WITH_PUBLIC_SEARCH_DOSSIER"
