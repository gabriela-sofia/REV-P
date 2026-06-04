import csv
import os
from pathlib import Path

import scripts.protocolo_c.revp_v2ae_common as common


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v2ae"


def write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    staging = tmp_path / "local_only" / "v2ae" / "staging"
    reports = tmp_path / "local_only" / "v2ae" / "reports"
    for p in (data, docs, cfg, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data


def _pkg(epc, event_id, region, patch_id, **over):
    row = {
        "event_patch_candidate_id": epc, "event_id": event_id, "event_region": region,
        "patch_id": patch_id, "patch_namespace": "EVENT_PATCH_CANDIDATE_NAMESPACE" if patch_id else "PATCH_ID_MISSING",
        "crosswalk_status": "EXPLICIT_DINO_CROSSWALK_NO_ANCHOR_CROSSWALK" if patch_id else "NO_CROSSWALK_PATCH_ID_MISSING",
        "sentinel_date_status": "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE" if patch_id else "SENTINEL_DATE_MISSING_WITH_BLOCKER",
        "date_linkability_status": "UNLINKABLE_NAMESPACE" if patch_id else "NO_DATE",
        "evidence_status": "CONTEXTUAL", "phenomenon_status": "flood_context",
        "coordinate_status": "NO_OCCURRENCE_COORD", "geometry_status": "NO_OBSERVED_GEOMETRY",
        "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED", "training_label_status": "BLOCKED",
        "dino_review_support_status": "DINO_REVIEW_SUPPORT_AVAILABLE",
        "blocker": "no_observed_geometry|no_occurrence_coordinates|unlinkable_sentinel_date|no_explicit_anchor_crosswalk|no_overlay|no_ground_reference|no_training_label|patch_truth_forbidden",
        "temporal_blocker": "sentinel_date_only_in_parallel_namespace_no_crosswalk",
        "package_validation_status": "PACKAGE_V2_SCHEMA_VALID_WITH_TEMPORAL_BLOCKER" if patch_id else "PACKAGE_V2_BLOCKED_MISSING_PATCH_ID",
        "schema_contract_version": "v2ab_event_patch_schema_contract",
        "can_create_ground_reference": "false", "can_create_training_label": "false",
        "ground_truth_operational": "false", "crosswalk_inferred": "false", "sentinel_date_inferred": "false",
    }
    row.update(over)
    return row


def install_base_inputs(data):
    write_csv(data / "v1uz_multiregion_closure_status.csv",
              ["closure_id", "region", "event_id", "closure_status", "best_evidence_type", "best_evidence_strength", "coordinate_status", "geometry_status", "overlay_status", "ground_reference_status", "main_blocker", "recommended_future_reopen_condition", "notes"],
              [
                  {"closure_id": "C0", "region": "REC", "event_id": "REC_2022_05_24_30", "closure_status": "RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL", "best_evidence_type": "CONTEXTUAL_COORDINATE_LAYER", "best_evidence_strength": "STRONG", "coordinate_status": "CONTEXTUAL_COORDINATE_ONLY_NO_OCCURRENCE", "geometry_status": "NO_OBSERVED_GEOMETRY", "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED", "main_blocker": "no_occurrence_coordinates", "recommended_future_reopen_condition": "x", "notes": "n"},
                  {"closure_id": "C1", "region": "PET", "event_id": "PET_2022_02_15", "closure_status": "PETROPOLIS_DOCUMENT_ONLY_NO_GEODATA", "best_evidence_type": "OFFICIAL_DOCUMENT_ONLY", "best_evidence_strength": "WEAK", "coordinate_status": "NO_COORDINATE_EVIDENCE", "geometry_status": "NO_OBSERVED_GEOMETRY", "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED", "main_blocker": "no_geodata", "recommended_future_reopen_condition": "x", "notes": "n"},
                  {"closure_id": "C2", "region": "PET", "event_id": "PET_2024_03_21_28", "closure_status": "PETROPOLIS_DOCUMENT_ONLY_NO_GEODATA", "best_evidence_type": "OFFICIAL_DOCUMENT_ONLY", "best_evidence_strength": "WEAK", "coordinate_status": "NO_COORDINATE_EVIDENCE", "geometry_status": "NO_OBSERVED_GEOMETRY", "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED", "main_blocker": "no_geodata", "recommended_future_reopen_condition": "x", "notes": "n"},
                  {"closure_id": "C3", "region": "CUR", "event_id": "CUR_2022_01_15", "closure_status": "CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL", "best_evidence_type": "PUBLIC_CONTEXT_LAYER_AND_HYDROMET", "best_evidence_strength": "MODERATE", "coordinate_status": "NO_OCCURRENCE_COORDINATE", "geometry_status": "NO_OBSERVED_GEOMETRY", "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED", "main_blocker": "no_occurrence_layer", "recommended_future_reopen_condition": "x", "notes": "n"},
              ])
    write_csv(data / "v2ad_qa_gate_summary.csv",
              ["gate_id", "qa_group", "total_checks", "passed_checks", "failed_checks", "expected_blockers", "gate_status", "required_action", "notes"],
              [{"gate_id": "G0", "qa_group": "OVERALL", "total_checks": "100", "passed_checks": "98", "failed_checks": "0", "expected_blockers": "2", "gate_status": "QA_PASS_WITH_EXPECTED_BLOCKERS", "required_action": "none", "notes": "n"}])
    packages = [
        _pkg("EPC0", "REC_2022_05_24_30", "REC", "REC_00001"),
        _pkg("EPC1", "PET_2022_02_15", "PET", "PET_00016"),
        _pkg("EPC2", "CUR_EVENT_REGISTRY_MISSING", "CUR", ""),
    ]
    write_csv(data / "v2ac_event_patch_v2_package_registry.csv", list(packages[0].keys()), packages)
    write_csv(data / "v2ac_schema_contract_validation.csv",
              ["schema_validation_id", "event_patch_candidate_id", "validation_status"],
              [{"schema_validation_id": f"SV{i}", "event_patch_candidate_id": p["event_patch_candidate_id"], "validation_status": "SCHEMA_VALID_NON_OPERATIONAL" if p["patch_id"] else "SCHEMA_INVALID_MISSING_PATCH_ID"} for i, p in enumerate(packages)])
    write_csv(data / "v1us_event_temporal_window_linkage.csv",
              ["temporal_linkage_id", "event_patch_candidate_id", "event_id", "patch_id", "region", "event_start_date", "event_end_date"],
              [{"temporal_linkage_id": "T0", "event_patch_candidate_id": "EPC0", "event_id": "REC_2022_05_24_30", "patch_id": "REC_00001", "region": "REC", "event_start_date": "2022-05-24", "event_end_date": "2022-05-30"},
               {"temporal_linkage_id": "T1", "event_patch_candidate_id": "EPC1", "event_id": "PET_2022_02_15", "patch_id": "PET_00016", "region": "PET", "event_start_date": "2022-02-15", "event_end_date": "2022-02-15"}])


def test_region_registry_contains_three_regions(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_canonical_region_registry_builder(common.parse_args([]))
    by_region = {r["region"]: r for r in rows}
    assert set(by_region) == {"REC", "PET", "CUR"}
    assert by_region["REC"]["canonical_region_status"] == "REGION_HARDENED_CONTEXTUAL_COORDINATE_NON_OPERATIONAL"
    assert by_region["PET"]["canonical_region_status"] == "REGION_HARDENED_DOCUMENT_ONLY_NO_GEODATA"
    assert by_region["CUR"]["canonical_region_status"] == "REGION_HARDENED_CONTEXT_ONLY_HOLD"
    assert all(r["overlay_status"] == "BLOCKED" for r in rows)
    assert all(r["ground_reference_status"] == "BLOCKED" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
