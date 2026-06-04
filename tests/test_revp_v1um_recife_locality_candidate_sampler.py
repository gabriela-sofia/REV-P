import csv
import os

import scripts.protocolo_c.revp_v1um_recife_common as common


def _write(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_base(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    data.mkdir(parents=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    router_cols = [
        "event_id", "candidate_row_id", "asset_id", "row_hash", "event_window_match",
        "hazard_signal", "coordinate_status", "locality_status", "sensitive_review_required",
        "review_route", "review_priority",
    ]
    routes = [
        {"event_id": common.EVENT_ID, "candidate_row_id": "c1", "asset_id": "a1", "row_hash": "rh1", "event_window_match": "event_core_window", "hazard_signal": "HAS_HAZARD_SIGNAL", "coordinate_status": "NO_COORDINATES", "locality_status": "ADDRESS_TEXT_AVAILABLE", "sensitive_review_required": "true", "review_route": common.LOCALITY_ROUTE, "review_priority": "2"},
        {"event_id": common.EVENT_ID, "candidate_row_id": "c2", "asset_id": "a1", "row_hash": "rh2", "event_window_match": "event_core_window", "hazard_signal": "HAS_HAZARD_SIGNAL", "coordinate_status": "NO_COORDINATES", "locality_status": "ADDRESS_TEXT_AVAILABLE", "sensitive_review_required": "true", "review_route": common.LOCALITY_ROUTE, "review_priority": "2"},
        {"event_id": common.EVENT_ID, "candidate_row_id": "c3", "asset_id": "a2", "row_hash": "rh3", "event_window_match": "event_core_window", "hazard_signal": "HAS_HAZARD_SIGNAL", "coordinate_status": "NO_COORDINATES", "locality_status": "ADDRESS_TEXT_AVAILABLE", "sensitive_review_required": "true", "review_route": common.LOCALITY_ROUTE, "review_priority": "2"},
    ]
    _write(data / "v1ul_recife_candidate_review_router.csv", router_cols, routes)
    match_cols = [
        "event_id", "asset_id", "row_hash", "window_type", "parsed_date",
        "has_flood_term", "has_rain_term", "has_landslide_term", "has_hazard_term",
        "has_neighborhood", "neighborhood_hash", "has_address", "address_hash",
        "has_coordinates", "coordinate_status", "candidate_status", "limitations",
    ]
    matches = [
        {"event_id": common.EVENT_ID, "asset_id": "a1", "row_hash": "rh1", "window_type": "event_core_window", "parsed_date": "2022-05-24", "has_flood_term": "true", "has_rain_term": "false", "has_landslide_term": "false", "has_hazard_term": "true", "has_neighborhood": "true", "neighborhood_hash": "n1hash", "has_address": "true", "address_hash": "addrhash1", "has_coordinates": "false", "coordinate_status": "NO_COORDINATES", "candidate_status": "candidate", "limitations": "synthetic"},
        {"event_id": common.EVENT_ID, "asset_id": "a1", "row_hash": "rh2", "window_type": "event_core_window", "parsed_date": "2022-05-25", "has_flood_term": "false", "has_rain_term": "true", "has_landslide_term": "false", "has_hazard_term": "true", "has_neighborhood": "true", "neighborhood_hash": "n2hash", "has_address": "true", "address_hash": "addrhash2", "has_coordinates": "false", "coordinate_status": "NO_COORDINATES", "candidate_status": "candidate", "limitations": "synthetic"},
        {"event_id": common.EVENT_ID, "asset_id": "a2", "row_hash": "rh3", "window_type": "event_core_window", "parsed_date": "2022-05-26", "has_flood_term": "false", "has_rain_term": "false", "has_landslide_term": "false", "has_hazard_term": "true", "has_neighborhood": "false", "neighborhood_hash": "", "has_address": "true", "address_hash": "addrhash3", "has_coordinates": "false", "coordinate_status": "NO_COORDINATES", "candidate_status": "candidate", "limitations": "synthetic"},
    ]
    _write(data / "v1uk_recife_event_window_match_registry.csv", match_cols, matches)
    _write(data / "v1uk_recife_candidate_row_registry.csv", ["candidate_row_id", "evidence_strength"], [{"candidate_row_id": "c1", "evidence_strength": "medium"}, {"candidate_row_id": "c2", "evidence_strength": "medium"}, {"candidate_row_id": "c3", "evidence_strength": "low"}])
    _write(data / "v1uk_recife_occurrence_table_profile.csv", ["asset_id", "table_name"], [{"asset_id": "a1", "table_name": "synthetic_table.csv"}, {"asset_id": "a2", "table_name": "synthetic_context.csv"}])
    _write(data / "v1uk_recife_asset_schema_registry.csv", ["asset_id", "title"], [{"asset_id": "a1", "title": "synthetic_table.csv"}, {"asset_id": "a2", "title": "synthetic_context.csv"}])
    return data


def test_sampler_creates_stratified_samples_without_literal_address(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    out = data / "sample.csv"
    rows = common.run_locality_candidate_sampler(str(out))
    sample_types = {r["sample_type"] for r in rows}
    content = out.read_text(encoding="utf-8")
    assert "high_confidence_sample" in sample_types
    assert "diversity_sample" in sample_types
    assert "Rua " not in content
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
