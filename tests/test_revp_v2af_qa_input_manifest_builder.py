import csv
import os

import scripts.protocolo_c.revp_v2af_common as common


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
    staging = tmp_path / "local_only" / "v2af" / "staging"
    reports = tmp_path / "local_only" / "v2af" / "reports"
    for p in (data, docs, cfg, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data


def _pkg(epc, event_id, region, patch_id):
    has = bool(patch_id)
    return {
        "event_patch_candidate_id": epc, "event_id": event_id, "event_region": region,
        "patch_id": patch_id,
        "patch_namespace": "EVENT_PATCH_CANDIDATE_NAMESPACE" if has else "PATCH_ID_MISSING",
        "crosswalk_status": "EXPLICIT_DINO_CROSSWALK_NO_ANCHOR_CROSSWALK" if has else "NO_CROSSWALK_PATCH_ID_MISSING",
        "anchor_patch_id": "", "refpatch_id": "",
        "sentinel_date_status": "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE" if has else "SENTINEL_DATE_MISSING_WITH_BLOCKER",
        "sentinel_scene_date": "", "date_linkability_status": "UNLINKABLE_NAMESPACE" if has else "NO_DATE",
        "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED", "training_label_status": "BLOCKED",
        "package_validation_status": "PACKAGE_V2_SCHEMA_VALID_WITH_TEMPORAL_BLOCKER" if has else "PACKAGE_V2_BLOCKED_MISSING_PATCH_ID",
        "can_create_ground_reference": "false", "can_create_training_label": "false",
        "ground_truth_operational": "false", "crosswalk_inferred": "false", "sentinel_date_inferred": "false",
    }


def _packages():
    rows = []
    # 171 valid (REC/PET split) + 1 missing-patch = 172.
    for i in range(115):
        rows.append(_pkg(f"EPC{i}", "PET_2022_02_15", "PET", f"PET_{i:05d}"))
    for i in range(56):
        rows.append(_pkg(f"EPCR{i}", "REC_2022_05_24_30", "REC", f"REC_{i:05d}"))
    rows.append(_pkg("EPCMISS", "CUR_EVENT_REGISTRY_MISSING", "CUR", ""))
    return rows


def install_base_inputs(data, packages=None):
    packages = packages if packages is not None else _packages()
    pkg_cols = list(packages[0].keys())
    write_csv(data / "v2ac_event_patch_v2_package_registry.csv", pkg_cols, packages)
    write_csv(data / "v2ac_schema_contract_validation.csv",
              ["schema_validation_id", "event_patch_candidate_id", "validation_status"],
              [{"schema_validation_id": f"SV{i}", "event_patch_candidate_id": p["event_patch_candidate_id"],
                "validation_status": "SCHEMA_VALID_NON_OPERATIONAL" if p["patch_id"] else "SCHEMA_INVALID_MISSING_PATCH_ID"} for i, p in enumerate(packages)])
    write_csv(data / "v2ac_migration_diff_audit.csv",
              ["diff_id", "event_patch_candidate_id", "migration_additive", "old_outputs_modified"],
              [{"diff_id": f"D{i}", "event_patch_candidate_id": p["event_patch_candidate_id"], "migration_additive": "true", "old_outputs_modified": "false"} for i, p in enumerate(packages)])
    # 2580 readiness rows = 172 * 15.
    rmatrix = []
    for p in packages:
        for d in range(15):
            rmatrix.append({"readiness_id": f"R{len(rmatrix)}", "event_patch_candidate_id": p["event_patch_candidate_id"], "dimension": f"dim{d}", "classification": "BLOCKED"})
    write_csv(data / "v2ac_v2_readiness_matrix.csv", ["readiness_id", "event_patch_candidate_id", "dimension", "classification"], rmatrix)
    # v2ad QA artifacts.
    for name in ("v2ad_package_contract_qa", "v2ad_namespace_crosswalk_qa", "v2ad_temporal_safety_qa",
                 "v2ad_guardrail_qa", "v2ad_readiness_consistency_qa", "v2ad_migration_integrity_qa"):
        write_csv(data / f"{name}.csv", ["qa_id", "status"], [{"qa_id": "Q0", "status": "PASS"}])
    write_csv(data / "v2ad_negative_fixture_qa.csv",
              ["negative_qa_id", "fixture_name", "status"],
              [{"negative_qa_id": f"N{i}", "fixture_name": f"f{i}", "status": "PASS_VIOLATION_DETECTED"} for i in range(10)])
    write_csv(data / "v2ad_qa_gate_summary.csv",
              ["gate_id", "qa_group", "gate_status"],
              [{"gate_id": "G0", "qa_group": "OVERALL", "gate_status": "QA_PASS_WITH_EXPECTED_BLOCKERS"}])
    # v2ae canonical registries.
    write_csv(data / "v2ae_canonical_region_registry.csv",
              ["region_registry_id", "region", "canonical_region_status", "overlay_status", "ground_reference_status"],
              [{"region_registry_id": f"RR{i}", "region": r, "canonical_region_status": s, "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED"}
               for i, (r, s) in enumerate(common.EXPECTED_REGION_STATUS.items())])
    write_csv(data / "v2ae_canonical_event_registry.csv",
              ["canonical_event_id", "event_id", "canonical_event_status"],
              [{"canonical_event_id": f"EV{i}", "event_id": e, "canonical_event_status": "EVENT_CANDIDATE_NON_OPERATIONAL"} for i, e in enumerate(common.EXPECTED_EVENTS)])
    write_csv(data / "v2ae_canonical_event_patch_registry.csv",
              ["canonical_package_id", "event_patch_candidate_id", "patch_namespace", "overlay_status", "ground_reference_status", "training_label_status"],
              [{"canonical_package_id": f"CP{i}", "event_patch_candidate_id": p["event_patch_candidate_id"], "patch_namespace": p["patch_namespace"], "overlay_status": "BLOCKED", "ground_reference_status": "BLOCKED", "training_label_status": "BLOCKED"} for i, p in enumerate(packages)])
    write_csv(data / "v2ae_multiregion_blocker_consolidation.csv", ["blocker_consolidation_id", "blocker"], [{"blocker_consolidation_id": "B0", "blocker": "no_observed_geometry"}])
    write_csv(data / "v2ae_multiregion_readiness_consolidation.csv", ["readiness_id", "status"], [{"readiness_id": "R0", "status": "BLOCKED"}])
    write_csv(data / "v2ae_region_reopen_condition_registry.csv",
              ["reopen_condition_id", "region"], [{"reopen_condition_id": f"RO{i}", "region": r} for i, r in enumerate(common.EXPECTED_REGIONS)])
    write_csv(data / "v2ae_safe_use_policy_registry.csv", ["policy_id", "scope"], [{"policy_id": "SU0", "scope": "GLOBAL"}])
    write_csv(data / "v2ae_registry_consistency_qa.csv", ["qa_id", "status"], [{"qa_id": "RCQA0", "status": "PASS"}])
    write_csv(data / "v2ae_next_programming_target_ranker.csv", ["rank", "next_target"], [{"rank": "1", "next_target": "EVENT_PATCH_PACKAGE_V2_QA_AUTOMATION"}])


def test_input_manifest_detects_missing_artifact(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_qa_input_manifest_builder(common.parse_args([]))
    assert all(r["existence_status"] != "MISSING" for r in rows)
    # Remove a required artifact -> manifest flags it MISSING.
    os.remove(str(data / "v2ae_canonical_region_registry.csv"))
    rows = common.run_qa_input_manifest_builder(common.parse_args([]))
    missing = [r for r in rows if r["artifact_path"].endswith("v2ae_canonical_region_registry.csv")]
    assert missing and missing[0]["existence_status"] == "MISSING"
    # No absolute paths in the manifest.
    assert all(not r["artifact_path"].startswith(("C:", "/")) for r in rows)
