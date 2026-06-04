import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.protocolo_c.revp_v1ul_recife_common as common


def _write(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_acceptance_audit_detects_complete_v1uk_outputs(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    data.mkdir(parents=True)
    docs.mkdir(parents=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))

    fixtures = {
        "v1uk_recife_asset_schema_registry.csv": (
            ["asset_id", "event_id", "row_count", "schema_status", "has_sensitive_fields"],
            [{"asset_id": "a1", "event_id": common.EVENT_ID, "row_count": "2", "schema_status": "SCHEMA_PROFILED", "has_sensitive_fields": "true"}],
        ),
        "v1uk_recife_field_semantics_registry.csv": (
            ["asset_id", "source_field", "canonical_field", "is_sensitive", "mapping_status"],
            [{"asset_id": "a1", "source_field": "bairro", "canonical_field": "neighborhood", "is_sensitive": "false", "mapping_status": "MAPPED"}],
        ),
        "v1uk_recife_occurrence_table_profile.csv": (
            ["asset_id", "total_rows", "rows_in_event_window", "rows_with_flood_terms", "rows_with_rain_terms", "rows_with_landslide_terms", "rows_with_coordinates", "rows_with_neighborhood", "rows_with_address"],
            [{"asset_id": "a1", "total_rows": "2", "rows_in_event_window": "1", "rows_with_flood_terms": "1", "rows_with_rain_terms": "0", "rows_with_landslide_terms": "0", "rows_with_coordinates": "0", "rows_with_neighborhood": "1", "rows_with_address": "1"}],
        ),
        "v1uk_recife_event_window_match_registry.csv": (
            ["event_id", "asset_id", "row_hash", "window_type", "has_hazard_term", "has_coordinates", "coordinate_status", "candidate_status"],
            [{"event_id": common.EVENT_ID, "asset_id": "a1", "row_hash": "rh", "window_type": "event_core_window", "has_hazard_term": "true", "has_coordinates": "false", "coordinate_status": "NO_COORDINATES", "candidate_status": "EVENT_WINDOW_OCCURRENCE_CANDIDATE_FOR_REVIEW"}],
        ),
        "v1uk_recife_coordinate_evidence_audit.csv": (
            ["asset_id", "coordinate_classification", "can_create_ground_reference", "can_create_training_label"],
            [{"asset_id": "a1", "coordinate_classification": "NO_COORDINATES", "can_create_ground_reference": "false", "can_create_training_label": "false"}],
        ),
        "v1uk_recife_locality_evidence_audit.csv": (
            ["asset_id", "locality_classification", "sufficient_for_human_review", "sufficient_for_overlay"],
            [{"asset_id": "a1", "locality_classification": "ADDRESS_TEXT_AVAILABLE", "sufficient_for_human_review": "true", "sufficient_for_overlay": "false"}],
        ),
        "v1uk_recife_candidate_row_registry.csv": (
            ["candidate_row_id", "event_id", "asset_id", "row_hash", "candidate_class", "event_window_match", "hazard_term_status", "coordinate_status", "locality_status", "can_create_ground_reference", "can_create_training_label", "required_next_action"],
            [{"candidate_row_id": "c1", "event_id": common.EVENT_ID, "asset_id": "a1", "row_hash": "rh", "candidate_class": "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW", "event_window_match": "event_core_window", "hazard_term_status": "HAS_HAZARD_SIGNAL", "coordinate_status": "NO_COORDINATES", "locality_status": "ADDRESS_TEXT_AVAILABLE", "can_create_ground_reference": "false", "can_create_training_label": "false", "required_next_action": "HUMAN_REVIEW"}],
        ),
        "v1uk_recife_supervisor_review_prepackage_registry.csv": (
            ["package_status", "candidate_rows_count", "coordinate_candidates_count", "locality_only_candidates_count"],
            [{"package_status": "READY", "candidate_rows_count": "1", "coordinate_candidates_count": "0", "locality_only_candidates_count": "1"}],
        ),
        "v1uk_recife_ground_reference_blocker_matrix.csv": (
            ["blocker", "status", "can_create_ground_reference", "can_create_training_label"],
            [{"blocker": "no_overlay", "status": "ACTIVE", "can_create_ground_reference": "false", "can_create_training_label": "false"}],
        ),
    }
    required = []
    required_cols = {}
    for name, (columns, rows) in fixtures.items():
        path = data / name
        _write(path, columns, rows)
        required.append(str(path))
        required_cols[str(path)] = columns
    report = docs / "protocolo_c_relatorio_v1uk_recife_ckan_schema_deep_audit.md"
    report.write_text("# report\n", encoding="utf-8")
    required.append(str(report))
    monkeypatch.setattr(common, "REQUIRED_V1UK_ARTIFACTS", required)
    monkeypatch.setattr(common, "V1UK_REQUIRED_COLUMNS", required_cols)

    rows = common.run_v1uk_acceptance_audit(str(data / "accept.csv"), str(docs / "audit.md"))
    summary = rows[-1]
    assert summary["status"] == "V1UK_COMPLETE"
    assert summary["exists"] == "true"
    assert "review_candidates=1" in summary["notes"]
