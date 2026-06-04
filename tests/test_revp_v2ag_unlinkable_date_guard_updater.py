import scripts.protocolo_c.revp_v2ag_common as common
from tests.test_revp_v2ag_crosswalk_source_inventory import install_packages, set_env


def test_unlinkable_guard_retains_prohibition_without_strong_crosswalk(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    common.run_crosswalk_source_inventory(common.parse_args([]))
    common.run_explicit_crosswalk_detector(common.parse_args([]))
    common.run_lineage_crosswalk_candidate_builder(common.parse_args([]))
    common.run_sentinel_date_linkability_auditor(common.parse_args([]))
    rows = common.run_unlinkable_date_guard_updater(common.parse_args([]))
    pet = [r for r in rows if r["patch_id"] == "PET_00002"][0]
    assert pet["new_guard_status"] != "EXPLICIT_CROSSWALK_FOUND_FUTURE_MIGRATION_ONLY"
    assert "apply_date_without_explicit_crosswalk" in pet["prohibited_use"]
    assert pet["sentinel_date_inferred"] == "false"
    assert pet["crosswalk_inferred"] == "false"
