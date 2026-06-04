import scripts.protocolo_c.revp_v2ag_common as common
from tests.test_revp_v2ag_crosswalk_source_inventory import install_packages, set_env, write_csv


def test_lineage_candidate_builder_records_weak_candidates_without_linking_dates(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    write_csv(data / "region_only.csv", ["event_patch_candidate_id", "patch_id", "region"], [{"event_patch_candidate_id": "EPC_SYN_002", "patch_id": "PET_00002", "region": "PET"}])
    common.run_crosswalk_source_inventory(common.parse_args([]))
    common.run_explicit_crosswalk_detector(common.parse_args([]))
    rows = common.run_lineage_crosswalk_candidate_builder(common.parse_args([]))
    rejected = [r for r in rows if r["lineage_evidence_type"] == "REGION_ONLY_REJECTED"]
    assert rejected
    assert all(r["accepted_as_explicit_crosswalk"] == "false" for r in rejected)
    assert all(r["can_link_sentinel_date"] == "false" for r in rejected)
    assert all(r["crosswalk_inferred"] == "false" for r in rows)
