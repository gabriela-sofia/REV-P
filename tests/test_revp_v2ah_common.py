import csv
from pathlib import Path

import pytest

import scripts.protocolo_c.revp_v2ah_common as common


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_env(tmp_path, monkeypatch):
    data = tmp_path / "datasets" / "protocolo_c"
    docs = tmp_path / "docs" / "metodologia_cientifica"
    cfg = tmp_path / "configs" / "protocolo_c"
    for p in (data, docs, cfg):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    return data


def install_inputs(data):
    package_cols = [
        "event_patch_candidate_id", "event_id", "event_region", "patch_id",
        "phenomenon_status", "evidence_status", "geometry_status",
        "coordinate_status", "ground_reference_status", "training_label_status",
        "dino_review_support_status", "crosswalk_status", "date_linkability_status",
        "blocker", "can_create_ground_reference", "can_create_training_label",
        "ground_truth_operational", "crosswalk_inferred", "sentinel_date_inferred",
    ]
    packages = [
        {
            "event_patch_candidate_id": "EPC_FIX_001", "event_id": "REC_2022_05_24_30",
            "event_region": "REC", "patch_id": "REC_00001",
            "phenomenon_status": "FLOOD_CONTEXT_REVIEW_ONLY",
            "evidence_status": "DOCUMENT_ONLY_NO_GEODATA",
            "geometry_status": "GEOMETRY_STILL_MISSING",
            "coordinate_status": "HYDROMET_CONTEXT_COORDINATE_ONLY",
            "ground_reference_status": "BLOCKED", "training_label_status": "BLOCKED",
            "dino_review_support_status": "DINO_REVIEW_SUPPORT_AVAILABLE",
            "crosswalk_status": "EXPLICIT_DINO_CROSSWALK_NO_ANCHOR_CROSSWALK",
            "date_linkability_status": "UNLINKABLE_NAMESPACE",
            "blocker": "unlinkable_sentinel_date|no_ground_reference",
            "can_create_ground_reference": "false", "can_create_training_label": "false",
            "ground_truth_operational": "false", "crosswalk_inferred": "false",
            "sentinel_date_inferred": "false",
        },
        {
            "event_patch_candidate_id": "EPC_FIX_002", "event_id": "PET_2022_02_15",
            "event_region": "PET", "patch_id": "PET_00002",
            "phenomenon_status": "MASS_MOVEMENT_CONTEXT_REVIEW_ONLY",
            "evidence_status": "DOCUMENT_ONLY_NO_GEODATA",
            "geometry_status": "GEOMETRY_STILL_MISSING",
            "coordinate_status": "NO_OCCURRENCE_COORDINATE",
            "ground_reference_status": "BLOCKED", "training_label_status": "BLOCKED",
            "dino_review_support_status": "", "crosswalk_status": "NO_EXPLICIT_ANCHOR_CROSSWALK",
            "date_linkability_status": "UNLINKABLE_NAMESPACE",
            "blocker": "no_observed_geometry",
            "can_create_ground_reference": "false", "can_create_training_label": "false",
            "ground_truth_operational": "false", "crosswalk_inferred": "false",
            "sentinel_date_inferred": "false",
        },
    ]
    write_csv(data / "v2ac_event_patch_v2_package_registry.csv", package_cols, packages)
    write_csv(data / "v2af_qa_gate_orchestration.csv", ["gate_id", "qa_component", "gate_status"], [{"gate_id": "GATE_v2af_OVERALL", "qa_component": "OVERALL", "gate_status": "QA_AUTOMATION_PASS_WITH_EXPECTED_BLOCKERS"}])
    write_csv(data / "v2ag_sentinel_date_linkability_audit.csv", ["event_patch_candidate_id", "patch_id", "linkability_status", "can_link_sentinel_date", "can_update_v2_package", "sentinel_date_inferred"], [{"event_patch_candidate_id": "EPC_FIX_001", "patch_id": "REC_00001", "linkability_status": "DATE_REMAINS_UNLINKABLE", "can_link_sentinel_date": "false", "can_update_v2_package": "false", "sentinel_date_inferred": "false"}])
    write_csv(data / "v2ag_unlinkable_date_guard_update.csv", ["event_patch_candidate_id", "patch_id", "future_allowed_action", "sentinel_date_inferred", "crosswalk_inferred"], [{"event_patch_candidate_id": "EPC_FIX_001", "patch_id": "REC_00001", "future_allowed_action": "manual_registry_lineage_review", "sentinel_date_inferred": "false", "crosswalk_inferred": "false"}])
    write_csv(data / "v2ag_next_programming_target_ranker.csv", ["rank", "next_target"], [{"rank": "1", "next_target": "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE"}])
    return packages


def test_common_fail_closed_bool_and_forbidden_promotion(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    assert common.bool_closed("TRUE") == "true"
    assert common.bool_closed("maybe") == "false"
    with pytest.raises(ValueError):
        common.assert_no_forbidden_promotion([{"ground_truth": "true"}])
    with pytest.raises(ValueError):
        common.assert_no_forbidden_promotion([{"source_path": r"C:\Users\gabriela\raw.tif"}])
