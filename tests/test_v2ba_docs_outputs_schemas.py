"""v2ba docs, public outputs and schemas tests."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_and_public_outputs_exist():
    docs = sorted((ROOT / "docs").glob("v2ba_*.md"))
    assert len(docs) == 4
    text = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    for term in ("REC_00019", "REC_2022_05_24_30", "CRS", "provenance", "C4_CANDIDATE_REQUIRES_HUMAN_REVIEW"):
        assert term.lower() in text.lower()
    assert (ROOT / "outputs_public" / "execution_reports" /
            "v2ba_minimal_real_geometry_acquisition_workbench_report.md").is_file()


def test_ten_schemas_are_valid_and_summary_is_tp0():
    schemas = sorted((ROOT / "datasets" / "schemas").glob("v2ba_*.schema.json"))
    assert len(schemas) == 10
    for path in schemas:
        obj = json.loads(path.read_text(encoding="utf-8"))
        assert obj["required"] and obj["properties"]
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2ba_minimal_real_geometry_acquisition_workbench_summary.json").read_text(encoding="utf-8"))
    assert summary["turning_point_level"] == "TP0_DOCUMENTED_ABSENCE_WITH_ACQUISITION_DOSSIER"
    assert summary["valid_patch_boundaries"] == 0
    assert summary["valid_event_polygons"] == 0
    assert summary["ready_pair_feed_rows"] == 0
