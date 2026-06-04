import csv
import os

import scripts.protocolo_c.revp_v1uo_multiregion_common as common


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
    _write(data / "event_candidate_registry.csv", [
        "event_id", "region", "city", "uf", "start_date", "end_date",
        "hazard_scope", "current_level", "current_status", "blocking_reason",
        "priority", "notes",
    ], [
        {"event_id": "PET_2022_02_15", "region": "PET", "city": "Petrópolis", "uf": "RJ", "start_date": "2022-02-15", "end_date": "2022-02-15", "hazard_scope": "mixed", "current_level": "C3", "current_status": "CANDIDATE", "blocking_reason": "phenomenon_separation_pending", "priority": "1", "notes": "synthetic"},
        {"event_id": "PET_2024_03_21_28", "region": "PET", "city": "Petrópolis", "uf": "RJ", "start_date": "2024-03-21", "end_date": "2024-03-28", "hazard_scope": "mixed", "current_level": "C2", "current_status": "CANDIDATE", "blocking_reason": "phenomenon_separation_pending", "priority": "2", "notes": "synthetic"},
        {"event_id": "REC_2022_05_24_30", "region": "REC", "city": "Recife", "uf": "PE", "start_date": "2022-05-24", "end_date": "2022-05-30", "hazard_scope": "urban_flooding", "current_level": "C3", "current_status": "CANDIDATE", "blocking_reason": "spatial_evidence_incomplete", "priority": "1", "notes": "synthetic"},
    ])
    _write(data / "v1un_recife_protocol_c_status_registry.csv", [
        "event_id", "new_status", "can_advance_to_overlay", "can_advance_to_ground_reference",
        "can_create_training_label",
    ], [{"event_id": "REC_2022_05_24_30", "new_status": "LOCALITY_ONLY_HUMAN_REVIEW_EVIDENCE_CONSOLIDATED", "can_advance_to_overlay": "false", "can_advance_to_ground_reference": "false", "can_create_training_label": "false"}])
    _write(data / "v1uf_event_hydromet_scorecard.csv", [
        "event_id", "has_temporal_anchor", "has_station_coordinates", "has_spatial_event_geometry",
        "has_phenomenon_separation",
    ], [
        {"event_id": "PET_2022_02_15", "has_temporal_anchor": "true", "has_station_coordinates": "true", "has_spatial_event_geometry": "false", "has_phenomenon_separation": "false"},
        {"event_id": "PET_2024_03_21_28", "has_temporal_anchor": "true", "has_station_coordinates": "true", "has_spatial_event_geometry": "false", "has_phenomenon_separation": "false"},
    ])
    _write(data / "v1ue_event_evidence_scorecard.csv", ["event_id", "classification"], [])
    _write(data / "v1uk_recife_asset_schema_registry.csv", ["asset_id"], [{"asset_id": "a1"}])
    _write(data / "v1uk_recife_occurrence_table_profile.csv", ["total_rows"], [{"total_rows": "10"}])
    return data


def test_event_registry_multiregion_preserves_recife_pet_and_curitiba_blocker(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    rows = common.run_multiregion_event_registry_builder(str(data / "events.csv"))
    by_event = {r["event_id"]: r for r in rows}
    assert by_event["REC_2022_05_24_30"]["current_best_evidence_status"] == "LOCALITY_ONLY_HUMAN_REVIEW_EVIDENCE_CONSOLIDATED"
    assert by_event["REC_2022_05_24_30"]["has_coordinate_evidence"] == "false"
    assert by_event["PET_2024_03_21_28"]["current_best_evidence_status"] == "TEMPORAL_HYDROMET_ANCHOR_CONFIRMED"
    assert by_event["PET_2022_02_15"]["main_blocker"] == "PHENOMENON_SEPARATION_REQUIRED"
    assert by_event["CUR_EVENT_REGISTRY_MISSING"]["current_best_evidence_status"] == "NEEDS_EVENT_REGISTRY_OR_PUBLIC_SOURCE_DEEPENING"
