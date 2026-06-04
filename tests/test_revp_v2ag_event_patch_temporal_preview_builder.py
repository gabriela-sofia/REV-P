import scripts.protocolo_c.revp_v2ag_common as common
from tests.test_revp_v2ag_crosswalk_source_inventory import install_packages, set_env, write_csv


def test_temporal_preview_never_applies_date_to_package(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    write_csv(data / "explicit.csv", ["event_patch_candidate_id", "patch_id", "refpatch_id"], [{"event_patch_candidate_id": "EPC_SYN_001", "patch_id": "REC_00001", "refpatch_id": "REFPATCH_SYN_DATE_001"}])
    common.run_crosswalk_source_inventory(common.parse_args([]))
    common.run_explicit_crosswalk_detector(common.parse_args([]))
    common.run_sentinel_date_linkability_auditor(common.parse_args([]))
    rows = common.run_event_patch_temporal_preview_builder(common.parse_args([]))
    assert any(r["preview_status"] == "PREVIEW_ONLY_NOT_APPLIED" for r in rows)
    assert all(r["applied_to_package"] == "false" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
