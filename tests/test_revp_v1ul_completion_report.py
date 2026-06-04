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


def test_completion_report_writes_locality_next_action_and_manifest(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    configs = tmp_path / "configs" / "protocolo_c"
    data.mkdir(parents=True)
    docs.mkdir(parents=True)
    configs.mkdir(parents=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(configs))
    monkeypatch.setattr(common, "V1UL_ARTIFACTS", [])
    monkeypatch.setattr(common, "REQUIRED_V1UK_ARTIFACTS", [])

    _write(data / "v1uk_recife_asset_schema_registry.csv", ["asset_id"], [{"asset_id": "a1"}])
    _write(data / "v1uk_recife_occurrence_table_profile.csv", [
        "total_rows", "rows_in_event_window", "rows_with_flood_terms",
        "rows_with_rain_terms", "rows_with_landslide_terms",
        "rows_with_neighborhood", "rows_with_address", "rows_with_coordinates",
    ], [{
        "total_rows": "1", "rows_in_event_window": "1", "rows_with_flood_terms": "1",
        "rows_with_rain_terms": "0", "rows_with_landslide_terms": "0",
        "rows_with_neighborhood": "1", "rows_with_address": "0", "rows_with_coordinates": "0",
    }])
    _write(data / "v1uk_recife_event_window_match_registry.csv", ["row_hash"], [{"row_hash": "rh"}])
    _write(data / "v1uk_recife_candidate_row_registry.csv", [
        "candidate_class"
    ], [{"candidate_class": "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW"}])
    _write(data / "v1ul_recife_candidate_review_router.csv", [
        "review_route", "can_enter_supervisor_review", "coordinate_status",
        "sensitive_review_required", "event_window_match", "hazard_signal",
    ], [{
        "review_route": "ROUTE_LOCALITY_ONLY_REVIEW",
        "can_enter_supervisor_review": "true",
        "coordinate_status": "NO_COORDINATES",
        "sensitive_review_required": "true",
        "event_window_match": "event_core_window",
        "hazard_signal": "HAS_HAZARD_SIGNAL",
    }])
    _write(data / "v1ul_recife_overlay_readiness_preflight.csv", [
        "overlay_preflight_status"
    ], [{"overlay_preflight_status": "LOCALITY_ONLY_NOT_OVERLAY_ELIGIBLE"}])
    _write(data / "v1ul_recife_candidate_decision_matrix.csv", [
        "decision_id"
    ], [{"decision_id": "d1"}])
    _write(data / "v1ul_recife_supervisor_review_queue.csv", [
        "queue_id"
    ], [{"queue_id": "q1"}])

    result = common.run_completion_report()
    assert result["locality_only_candidates"] == 1
    assert result["next_action"] == "v1um - Recife Locality-Only Human Review Package"
    assert (docs / "protocolo_c_status_atual_v1ul.md").exists()
    assert (data / "v1ul_versionable_artifacts_manifest.csv").exists()
