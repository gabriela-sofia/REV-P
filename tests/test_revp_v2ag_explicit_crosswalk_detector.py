import scripts.protocolo_c.revp_v2ag_common as common
from tests.test_revp_v2ag_crosswalk_source_inventory import install_packages, set_env, write_csv


def test_explicit_detector_accepts_same_row_refpatch_and_rejects_region_only(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    write_csv(data / "explicit.csv", ["event_patch_candidate_id", "patch_id", "refpatch_id", "region"], [{"event_patch_candidate_id": "EPC_SYN_001", "patch_id": "REC_00001", "refpatch_id": "REFPATCH_SYN_DATE_001", "region": "REC"}])
    write_csv(data / "region_only.csv", ["event_patch_candidate_id", "patch_id", "region"], [{"event_patch_candidate_id": "EPC_SYN_002", "patch_id": "PET_00002", "region": "PET"}])
    common.run_crosswalk_source_inventory(common.parse_args([]))
    rows = common.run_explicit_crosswalk_detector(common.parse_args([]))
    strong = [r for r in rows if r["event_patch_patch_id"] == "REC_00001" and r["crosswalk_type"] == "PATCH_TO_REFPATCH_EXPLICIT"]
    assert strong and strong[0]["can_link_sentinel_date"] == "true"
    assert not any(r["event_patch_patch_id"] == "PET_00002" and r["crosswalk_type"] != "PATCH_TO_DINO_EXPLICIT" for r in rows)
    assert all(r["crosswalk_inferred"] == "false" for r in rows)
