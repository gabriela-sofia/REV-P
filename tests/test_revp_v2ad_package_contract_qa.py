import csv
import os
from pathlib import Path

import scripts.protocolo_c.revp_v2ad_common as common


REAL_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v2ad"

REQUIRED = common.REQUIRED_CONTRACT_DEFAULT
OPTIONAL = ["sentinel_scene_date", "sentinel_scene_datetime", "sentinel_platform",
            "scene_id", "source_patch_id", "anchor_patch_id", "refpatch_id", "explicit_crosswalk_id"]


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
    staging = tmp_path / "local_only" / "v2ad" / "staging"
    reports = tmp_path / "local_only" / "v2ad" / "reports"
    for p in (data, docs, cfg, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "NEGATIVE_FIXTURE_DIR", str(REAL_FIXTURE_DIR))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data


def pkg(**over):
    row = {
        "event_patch_candidate_id": "EPC0", "event_id": "REC_2022_05_24_30",
        "event_region": "REC", "patch_id": "REC_00001",
        "patch_namespace": common.NS_EVENT, "patch_source_registry": "v1us_patch_registry_resolution.csv",
        "source_patch_id": "REC_00001", "anchor_patch_id": "", "refpatch_id": "",
        "explicit_crosswalk_id": "XW_DINO::REC_00001",
        "crosswalk_status": "EXPLICIT_DINO_CROSSWALK_NO_ANCHOR_CROSSWALK",
        "linkage_basis": "REGION_ONLY_CANDIDATE", "linkage_status": "CANDIDATE_NON_OPERATIONAL",
        "event_patch_candidate_only": "true", "sentinel_scene_date": "",
        "sentinel_scene_datetime": "", "sentinel_platform": "", "scene_id": "",
        "sentinel_date_status": "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE",
        "date_source_namespace": "ANCHOR_REFPATCH_NAMESPACE",
        "date_linkability_status": "UNLINKABLE_NAMESPACE",
        "temporal_linkage_status": "BLOCKED_NO_LINKABLE_DATE",
        "temporal_blocker": "sentinel_date_only_in_parallel_namespace_no_crosswalk",
        "evidence_status": "CONTEXTUAL", "phenomenon_status": "flood_context",
        "coordinate_status": "NO_OCCURRENCE_COORD", "geometry_status": "NO_OBSERVED_GEOMETRY",
        "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED",
        "training_label_status": "BLOCKED", "dino_review_support_status": "DINO_REVIEW_SUPPORT_AVAILABLE",
        "blocker": "no_observed_geometry|unlinkable_sentinel_date|no_explicit_anchor_crosswalk|no_overlay|no_ground_reference|no_training_label|patch_truth_forbidden",
        "safe_use": "contextual_review_only", "prohibited_use": "ground_truth_label_overlay_patch_truth",
        "schema_contract_version": "v2ab_event_patch_schema_contract",
        "package_validation_status": "PACKAGE_V2_SCHEMA_VALID_WITH_TEMPORAL_BLOCKER",
        "can_create_ground_reference": "false", "can_create_training_label": "false",
        "ground_truth_operational": "false", "crosswalk_inferred": "false",
        "sentinel_date_inferred": "false",
    }
    row.update(over)
    return row


PKG_COLUMNS = list(pkg().keys())


def missing_pkg():
    return pkg(event_patch_candidate_id="EPC2", event_id="CUR_EVENT_REGISTRY_MISSING",
              event_region="CUR", patch_id="", patch_namespace="PATCH_ID_MISSING",
              source_patch_id="", explicit_crosswalk_id="",
              crosswalk_status="NO_CROSSWALK_PATCH_ID_MISSING",
              sentinel_date_status="SENTINEL_DATE_MISSING_WITH_BLOCKER",
              date_source_namespace="", date_linkability_status="NO_DATE",
              temporal_blocker="no_recoverable_sentinel_date_for_this_patch",
              package_validation_status="PACKAGE_V2_BLOCKED_MISSING_PATCH_ID")


def install_base_inputs(data, packages=None):
    packages = packages if packages is not None else [pkg(), missing_pkg()]
    write_csv(data / "v2ac_event_patch_v2_package_registry.csv", PKG_COLUMNS, packages)
    # v2ab contract
    rows = [{"contract_field_id": f"CF{i}", "field_name": f, "field_group": "g", "required": "true", "nullable": "false", "requires_blocker_if_null": "true", "allowed_values": "", "forbidden_values": "", "description": "d", "notes": "n"} for i, f in enumerate(REQUIRED)]
    rows += [{"contract_field_id": f"CFO{i}", "field_name": f, "field_group": "g", "required": "false", "nullable": "true", "requires_blocker_if_null": "true", "allowed_values": "", "forbidden_values": "patch_date_inferred", "description": "d", "notes": "n"} for i, f in enumerate(OPTIONAL)]
    write_csv(data / "v2ab_event_patch_schema_contract.csv",
              ["contract_field_id", "field_name", "field_group", "required", "nullable", "requires_blocker_if_null", "allowed_values", "forbidden_values", "description", "notes"], rows)
    # readiness matrix consistent with packages
    dims = {
        "event_identity": "STRONG", "patch_identity": "STRONG", "patch_namespace": "STRONG",
        "explicit_crosswalk": "MODERATE", "sentinel_date_status": "BLOCKED", "temporal_linkage": "BLOCKED",
        "evidence_attachment": "MODERATE", "phenomenon_support": "MODERATE", "coordinate_support": "ABSENT",
        "geometry_support": "ABSENT", "dino_review_support": "MODERATE", "overlay_readiness": "BLOCKED",
        "ground_reference_readiness": "BLOCKED", "training_readiness": "BLOCKED", "package_schema_validity": "STRONG",
    }
    rmatrix = []
    for p in packages:
        for dim, cls in dims.items():
            c = cls
            if p["patch_id"] == "" and dim == "patch_identity":
                c = "ABSENT"
            rmatrix.append({"readiness_id": f"R{len(rmatrix)}", "event_patch_candidate_id": p["event_patch_candidate_id"],
                            "event_id": p["event_id"], "patch_id": p["patch_id"], "dimension": dim, "classification": c,
                            "basis": "b", "ground_truth_operational": "false", "can_create_ground_reference": "false",
                            "can_create_training_label": "false", "can_reopen_protocol_b": "false", "dino_usage": "SUPPORT_ONLY",
                            "no_overlay_executed": "true", "no_coordinates_invented": "true", "patch_bound_truth": "false",
                            "operational_validation": "false", "schema_migration_only": "true", "crosswalk_inferred": "false",
                            "sentinel_date_inferred": "false", "raw_data_versioned": "false", "notes": "n"})
    write_csv(data / "v2ac_v2_readiness_matrix.csv", list(rmatrix[0].keys()), rmatrix)
    # migration diff (additive)
    write_csv(data / "v2ac_migration_diff_audit.csv",
              ["diff_id", "event_patch_candidate_id", "source_version", "target_version", "fields_added", "statuses_changed", "blockers_added", "old_outputs_modified", "migration_additive", "notes"],
              [{"diff_id": f"D{i}", "event_patch_candidate_id": p["event_patch_candidate_id"], "source_version": "v1us", "target_version": "v2ac", "fields_added": "patch_namespace|crosswalk_status", "statuses_changed": "crosswalk_status", "blockers_added": "no_explicit_anchor_crosswalk", "old_outputs_modified": "false", "migration_additive": "true", "notes": "n"} for i, p in enumerate(packages)])
    # v1us candidate registry (ids preserved)
    write_csv(data / "v1us_event_patch_candidate_registry.csv",
              ["event_patch_candidate_id", "event_id", "region", "patch_id", "blocker"],
              [{"event_patch_candidate_id": p["event_patch_candidate_id"], "event_id": p["event_id"], "region": p["event_region"], "patch_id": p["patch_id"], "blocker": "b"} for p in packages])


def test_contract_qa_passes_clean_and_detects_missing_field(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_package_contract_qa(common.parse_args([]))
    assert sum(1 for r in rows if r["status"] == "FAIL") == 0
    # Missing-patch package is an expected blocker, not a failure.
    assert any(r["status"] == "EXPECTED_BLOCKER" for r in rows)

    # Tamper: drop schema_contract_version on the valid package -> FAIL.
    bad = pkg(schema_contract_version="")
    install_base_inputs(data, packages=[bad, missing_pkg()])
    rows = common.run_package_contract_qa(common.parse_args([]))
    assert any(r["check_name"] == "schema_contract_version_set" and r["status"] == "FAIL" for r in rows)
