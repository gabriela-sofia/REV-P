import csv
import os
from pathlib import Path

import scripts.protocolo_c.revp_v2ac_common as common


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "v2ac"

REQUIRED_CONTRACT_FIELDS = [
    "event_patch_candidate_id", "event_id", "event_region", "patch_id",
    "patch_namespace", "patch_source_registry", "linkage_basis", "linkage_status",
    "event_patch_candidate_only", "sentinel_date_status", "temporal_linkage_status",
    "evidence_status", "geometry_status", "overlay_status", "ground_reference_status",
    "training_label_status", "blocker", "safe_use", "prohibited_use",
]
OPTIONAL_CONTRACT_FIELDS = [
    "sentinel_scene_date", "sentinel_scene_datetime", "sentinel_platform",
    "scene_id", "source_patch_id", "anchor_patch_id", "refpatch_id",
    "explicit_crosswalk_id",
]


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
    staging = tmp_path / "local_only" / "v2ac" / "staging"
    reports = tmp_path / "local_only" / "v2ac" / "reports"
    for p in (data, docs, cfg, staging, reports):
        p.mkdir(parents=True, exist_ok=True)
    dino = data / "dino_patch_visual_linkage_fixture.csv"
    import shutil
    shutil.copy(FIXTURE_DIR / "dino_patch_visual_linkage_fixture.csv", dino)
    monkeypatch.setattr(common, "DATASET_DIR", str(data))
    monkeypatch.setattr(common, "DOCS_DIR", str(docs))
    monkeypatch.setattr(common, "CONFIG_DIR", str(cfg))
    monkeypatch.setattr(common, "DINO_VISUAL_REGISTRY", str(dino))
    monkeypatch.setattr(common, "STAGING_DIR", str(staging))
    monkeypatch.setattr(common, "REPORTS_DIR", str(reports))
    return data


def install_base_inputs(data):
    write_csv(data / "v1us_event_patch_candidate_registry.csv",
              ["event_patch_candidate_id", "event_id", "region", "patch_id", "linkage_basis", "linkage_status", "event_patch_candidate_only", "patch_bound_truth", "can_create_ground_reference", "can_create_training_label", "blocker", "notes"],
              [
                  {"event_patch_candidate_id": "EPC0", "event_id": "REC_2022_05_24_30", "region": "REC", "patch_id": "REC_00001", "linkage_basis": "REGION_ONLY_CANDIDATE", "linkage_status": "CANDIDATE_NON_OPERATIONAL", "event_patch_candidate_only": "true", "patch_bound_truth": "false", "can_create_ground_reference": "false", "can_create_training_label": "false", "blocker": "SENTINEL_DATE_AND_GEOMETRY_MISSING", "notes": "x"},
                  {"event_patch_candidate_id": "EPC1", "event_id": "PET_2022_02_15", "region": "PET", "patch_id": "PET_00016", "linkage_basis": "REGION_ONLY_CANDIDATE", "linkage_status": "CANDIDATE_NON_OPERATIONAL", "event_patch_candidate_only": "true", "patch_bound_truth": "false", "can_create_ground_reference": "false", "can_create_training_label": "false", "blocker": "SENTINEL_DATE_AND_GEOMETRY_MISSING", "notes": "x"},
                  {"event_patch_candidate_id": "EPC2", "event_id": "CUR_EVENT_REGISTRY_MISSING", "region": "CUR", "patch_id": "", "linkage_basis": "EVENT_REGISTRY_MISSING", "linkage_status": "BLOCKED_NO_CLEAR_EVENT", "event_patch_candidate_only": "true", "patch_bound_truth": "false", "can_create_ground_reference": "false", "can_create_training_label": "false", "blocker": "CURITIBA_EVENT_REGISTRY_MISSING", "notes": "x"},
                  {"event_patch_candidate_id": "EPC3", "event_id": "REC_2022_05_24_30", "region": "REC", "patch_id": "REC_00009", "linkage_basis": "REGION_ONLY_CANDIDATE", "linkage_status": "CANDIDATE_NON_OPERATIONAL", "event_patch_candidate_only": "true", "patch_bound_truth": "false", "can_create_ground_reference": "false", "can_create_training_label": "false", "blocker": "GEOMETRY_MISSING", "notes": "x"},
              ])
    write_csv(data / "v2ab_temporal_field_contract_enforcement.csv",
              ["temporal_contract_id", "event_patch_candidate_id", "event_id", "patch_id", "patch_namespace", "sentinel_date_status", "selected_sentinel_date", "date_source_namespace", "date_linkability_status", "temporal_blocker", "sentinel_date_inferred", "notes"],
              [
                  {"temporal_contract_id": "TC0", "event_patch_candidate_id": "EPC0", "event_id": "REC_2022_05_24_30", "patch_id": "REC_00001", "patch_namespace": common.NS_EVENT, "sentinel_date_status": "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE", "selected_sentinel_date": "", "date_source_namespace": "ANCHOR_REFPATCH_NAMESPACE", "date_linkability_status": "NOT_LINKABLE_DIFFERENT_NAMESPACE_NO_EXPLICIT_CROSSWALK", "temporal_blocker": "sentinel_date_only_in_parallel_namespace_no_crosswalk", "sentinel_date_inferred": "false", "notes": "x"},
                  {"temporal_contract_id": "TC1", "event_patch_candidate_id": "EPC1", "event_id": "PET_2022_02_15", "patch_id": "PET_00016", "patch_namespace": common.NS_EVENT, "sentinel_date_status": "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE", "selected_sentinel_date": "", "date_source_namespace": "ANCHOR_REFPATCH_NAMESPACE", "date_linkability_status": "NOT_LINKABLE_DIFFERENT_NAMESPACE_NO_EXPLICIT_CROSSWALK", "temporal_blocker": "sentinel_date_only_in_parallel_namespace_no_crosswalk", "sentinel_date_inferred": "false", "notes": "x"},
                  {"temporal_contract_id": "TC2", "event_patch_candidate_id": "EPC2", "event_id": "CUR_EVENT_REGISTRY_MISSING", "patch_id": "", "patch_namespace": "", "sentinel_date_status": "SENTINEL_DATE_MISSING_WITH_BLOCKER", "selected_sentinel_date": "", "date_source_namespace": "", "date_linkability_status": "NOT_LINKABLE_NO_DATE", "temporal_blocker": "no_recoverable_sentinel_date_for_this_patch", "sentinel_date_inferred": "false", "notes": "x"},
                  {"temporal_contract_id": "TC3", "event_patch_candidate_id": "EPC3", "event_id": "REC_2022_05_24_30", "patch_id": "REC_00009", "patch_namespace": common.NS_EVENT, "sentinel_date_status": "SENTINEL_DATE_CONFIRMED_SAME_PATCH", "selected_sentinel_date": "2022-05-25", "date_source_namespace": common.NS_EVENT, "date_linkability_status": "LINKABLE_SAME_PATCH", "temporal_blocker": "", "sentinel_date_inferred": "false", "notes": "x"},
              ])
    epcs = ["EPC0", "EPC1", "EPC3"]
    write_csv(data / "v1us_phenomenon_status_attachment.csv",
              ["phenomenon_attachment_id", "event_patch_candidate_id", "event_id", "patch_id", "region", "phenomenon_class"],
              [{"phenomenon_attachment_id": f"PH{i}", "event_patch_candidate_id": e, "event_id": "E", "patch_id": "P", "region": "REC", "phenomenon_class": "flood_context"} for i, e in enumerate(epcs)])
    write_csv(data / "v1us_geometry_blocker_attachment.csv",
              ["geometry_blocker_id", "event_patch_candidate_id", "event_id", "patch_id", "region", "coordinate_status", "geometry_status", "overlay_blocker"],
              [{"geometry_blocker_id": f"G{i}", "event_patch_candidate_id": e, "event_id": "E", "patch_id": "P", "region": "REC", "coordinate_status": "NO_OCCURRENCE_COORD", "geometry_status": "NO_OBSERVED_GEOMETRY", "overlay_blocker": "BLOCKED"} for i, e in enumerate(epcs)])
    write_csv(data / "v1us_external_evidence_attachment_registry.csv",
              ["attachment_id", "event_patch_candidate_id", "event_id", "patch_id", "evidence_source", "evidence_status"],
              [{"attachment_id": f"A{i}", "event_patch_candidate_id": e, "event_id": "E", "patch_id": "P", "evidence_source": "official", "evidence_status": "CONTEXTUAL"} for i, e in enumerate(epcs)])
    write_csv(data / "v1us_dino_review_support_attachment.csv",
              ["dino_attachment_id", "event_patch_candidate_id", "event_id", "patch_id", "region", "dino_review_support_status", "dino_usage"],
              [{"dino_attachment_id": f"D{i}", "event_patch_candidate_id": e, "event_id": "E", "patch_id": "P", "region": "REC", "dino_review_support_status": "DINO_REVIEW_SUPPORT_AVAILABLE", "dino_usage": "SUPPORT_ONLY"} for i, e in enumerate(epcs)])
    # Minimal v2ab schema contract (required + optional field names).
    rows = [{"contract_field_id": f"CF{i}", "field_name": f, "field_group": "g", "required": "true", "nullable": "false", "requires_blocker_if_null": "true", "allowed_values": "", "forbidden_values": "", "description": "d", "notes": "n"} for i, f in enumerate(REQUIRED_CONTRACT_FIELDS)]
    rows += [{"contract_field_id": f"CFO{i}", "field_name": f, "field_group": "g", "required": "false", "nullable": "true", "requires_blocker_if_null": "true", "allowed_values": "", "forbidden_values": "patch_date_inferred", "description": "d", "notes": "n"} for i, f in enumerate(OPTIONAL_CONTRACT_FIELDS)]
    write_csv(data / "v2ab_event_patch_schema_contract.csv",
              ["contract_field_id", "field_name", "field_group", "required", "nullable", "requires_blocker_if_null", "allowed_values", "forbidden_values", "description", "notes"], rows)


def test_v2_package_registry_preserves_all_packages(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    rows = common.run_event_patch_v2_package_builder(common.parse_args([]))
    assert len(rows) == 4  # all candidates preserved
    epcs = {r["event_patch_candidate_id"] for r in rows}
    assert epcs == {"EPC0", "EPC1", "EPC2", "EPC3"}
    for r in rows:
        # v2 schema fields present.
        for field in ("patch_namespace", "patch_source_registry", "crosswalk_status",
                      "sentinel_date_status", "date_linkability_status", "schema_contract_version"):
            assert field in r
        assert r["overlay_status"] == "BLOCKED"
        assert r["ground_reference_status"] == "BLOCKED"
        assert r["training_label_status"] == "BLOCKED"
        assert r["can_create_ground_reference"] == "false"
        assert r["crosswalk_inferred"] == "false"
        assert r["sentinel_date_inferred"] == "false"
