"""v2ay - docs, public outputs and schemas tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_docs_exist_with_turning_point_terms():
    names = [
        "v2ay_event_scope_reconciliation.md", "v2ay_turning_point_definition.md",
        "v2ay_real_geometry_acquisition_playbook.md", "v2ay_current_scientific_state.md",
    ]
    combined = ""
    for name in names:
        path = ROOT / "docs" / name
        assert path.is_file()
        combined += path.read_text(encoding="utf-8")
    for term in ("TP0", "TP1", "TP2", "TP3", "TP4", "CRS", "Petropolis", "ground truth"):
        assert term.lower() in combined.lower()


def test_ten_v2ay_schemas_are_valid_json():
    paths = sorted((ROOT / "datasets" / "schemas").glob("v2ay_*.schema.json"))
    assert len(paths) >= 10
    new_paths = [path for path in paths if path.name in {
        "v2ay_region_event_canonical_registry.schema.json",
        "v2ay_event_scope_reconciliation_audit.schema.json",
        "v2ay_event_package_patch_crosswalk.schema.json",
        "v2ay_geometry_gap_analysis.schema.json",
        "v2ay_minimum_real_geometry_contract.schema.json",
        "v2ay_geometry_acquisition_targets.schema.json",
        "v2ay_external_source_query_plan.schema.json",
        "v2ay_spatial_metadata_absence_certificate.schema.json",
        "v2ay_pipeline_replay_plan.schema.json",
        "v2ay_turning_point_readiness_gate.schema.json",
    }]
    assert len(new_paths) == 10
    for path in new_paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["required"] and schema["properties"]


def test_public_report_summary_and_log_exist():
    assert (ROOT / "outputs_public" / "execution_reports" /
            "v2ay_event_scope_reconciliation_turning_point_report.md").is_file()
    assert (ROOT / "outputs_public" / "execution_reports" /
            "v2ay_event_scope_reconciliation_turning_point_summary.json").is_file()
    assert (ROOT / "outputs_public" / "logs_summary" /
            "v2ay_event_scope_reconciliation_turning_point.txt").is_file()
