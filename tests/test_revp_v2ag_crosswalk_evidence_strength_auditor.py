import scripts.protocolo_c.revp_v2ag_common as common
from tests.test_revp_v2ag_crosswalk_source_inventory import install_packages, set_env, write_csv


def test_evidence_strength_only_enables_linkability_for_strong_explicit(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    write_csv(data / "explicit.csv", ["event_patch_candidate_id", "patch_id", "refpatch_id"], [{"event_patch_candidate_id": "EPC_SYN_001", "patch_id": "REC_00001", "refpatch_id": "REFPATCH_SYN_DATE_001"}])
    write_csv(data / "dino.csv", ["event_patch_candidate_id", "patch_id", "dino_patch_id"], [{"event_patch_candidate_id": "EPC_SYN_002", "patch_id": "PET_00002", "dino_patch_id": "DINO_SYN_002"}])
    common.run_crosswalk_source_inventory(common.parse_args([]))
    common.run_explicit_crosswalk_detector(common.parse_args([]))
    common.run_lineage_crosswalk_candidate_builder(common.parse_args([]))
    rows = common.run_crosswalk_evidence_strength_auditor(common.parse_args([]))
    assert any(r["evidence_class"] == "STRONG_EXPLICIT_CROSSWALK" and r["can_enable_date_linkability"] == "true" for r in rows)
    assert all(r["can_update_package_v2"] == "false" for r in rows)
    dino_rows = [r for r in rows if r["blocker"] == "DINO_CROSSWALK_NOT_SENTINEL_DATE_CROSSWALK"]
    assert dino_rows and all(r["can_enable_date_linkability"] == "false" for r in dino_rows)
