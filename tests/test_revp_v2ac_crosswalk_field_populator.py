from tests.test_revp_v2ac_event_patch_v2_package_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ac_common as common


def test_crosswalk_populator_only_explicit_dino_no_anchor(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_event_patch_v2_package_builder(common.parse_args([]))
    rows = common.run_crosswalk_field_populator(common.parse_args([]))
    by_epc = {r["event_patch_candidate_id"]: r for r in rows}
    # REC_00001 and PET_00016 are in the DINO set -> explicit DINO crosswalk.
    assert by_epc["EPC0"]["crosswalk_status"].startswith("EXPLICIT_DINO")
    assert by_epc["EPC0"]["dino_patch_id"] == "REC_00001"
    assert by_epc["EPC0"]["explicit_crosswalk_id"] == "XW_DINO::REC_00001"
    # REC_00009 is NOT in the DINO set -> no explicit crosswalk.
    assert by_epc["EPC3"]["crosswalk_status"] == "NO_EXPLICIT_CROSSWALK"
    assert by_epc["EPC3"]["dino_patch_id"] == ""
    # Anchor / refpatch crosswalk is NEVER populated, and never inferred.
    for r in rows:
        assert r["anchor_patch_id"] == ""
        assert r["refpatch_id"] == ""
        assert r["crosswalk_inferred"] == "false"
