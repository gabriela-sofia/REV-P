"""v2bc docs, schemas and summary tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_and_seven_schemas_exist():
    assert len(list((ROOT / "docs").glob("v2bc_*.md"))) == 4
    schemas = list((ROOT / "datasets" / "schemas").glob("v2bc_*.schema.json"))
    assert len(schemas) == 7
    assert all(json.loads(p.read_text(encoding="utf-8"))["required"] for p in schemas)


def test_summary_is_contextual_tp0():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2bc_recife_gis_digitization_workbench_summary.json").read_text(encoding="utf-8"))
    assert summary["risk_context_features"] == 400
    assert summary["operational_patch_boundaries"] == summary["operational_event_polygons"] == 0
    assert summary["ready_feeds_created"] == 0
    assert summary["turning_point_level"] == "TP0_CONTEXTUAL_GIS_WORKBENCH_READY"
