import csv
import os

import scripts.protocolo_c.revp_v1uk_recife_common as common


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_completion_report_blocks_ground_reference_and_writes_docs(tmp_path, monkeypatch):
    data = tmp_path / "datasets"
    docs = tmp_path / "docs"
    data.mkdir()
    docs.mkdir()
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "V1UK_ARTIFACTS", [])
    _write(data / "v1uk_recife_asset_schema_registry.csv", ["asset_id"], [{"asset_id": "a"}])
    _write(data / "v1uk_recife_occurrence_table_profile.csv",
           ["total_rows", "rows_in_event_window", "rows_with_flood_terms", "rows_with_rain_terms", "rows_with_landslide_terms", "rows_with_neighborhood", "rows_with_address", "rows_with_coordinates"],
           [{"total_rows": "2", "rows_in_event_window": "1", "rows_with_flood_terms": "1", "rows_with_rain_terms": "0", "rows_with_landslide_terms": "0", "rows_with_neighborhood": "1", "rows_with_address": "1", "rows_with_coordinates": "0"}])
    _write(data / "v1uk_recife_event_window_match_registry.csv", ["row_hash"], [{"row_hash": "r"}])
    _write(data / "v1uk_recife_coordinate_evidence_audit.csv", ["asset_id", "coordinate_classification"], [{"asset_id": "ec18759d-fac2-445e-ae72-af9d9210b831", "coordinate_classification": "REGIONAL_CONTEXT_POINTS"}])
    _write(data / "v1uk_recife_locality_evidence_audit.csv", ["asset_id"], [{"asset_id": "a"}])
    _write(data / "v1uk_recife_candidate_row_registry.csv", ["candidate_class"], [{"candidate_class": "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW"}])
    _write(data / "v1uk_recife_supervisor_review_prepackage_registry.csv", ["package_status"], [{"package_status": "READY"}])
    result = common.run_completion_report()
    assert result["locality_candidates"] == 1
    blocker = data / "v1uk_recife_ground_reference_blocker_matrix.csv"
    content = blocker.read_text(encoding="utf-8")
    assert "can_create_ground_reference,true" not in content
    assert os.path.exists(docs / "protocolo_c_status_atual_v1uk.md")
