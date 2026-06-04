from tests.test_revp_v2ac_event_patch_v2_package_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ac_common as common


def test_temporal_populator_never_fills_unlinkable_date(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_event_patch_v2_package_builder(common.parse_args([]))
    rows = common.run_temporal_status_field_populator(common.parse_args([]))
    by_epc = {r["event_patch_candidate_id"]: r for r in rows}
    # Unlinkable: status preserved but sentinel_scene_date stays empty.
    assert by_epc["EPC0"]["sentinel_date_status"] == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE"
    assert by_epc["EPC0"]["sentinel_scene_date"] == ""
    assert by_epc["EPC0"]["date_linkability_status"] == "UNLINKABLE_NAMESPACE"
    # Missing: no date.
    assert by_epc["EPC2"]["sentinel_date_status"] == "SENTINEL_DATE_MISSING_WITH_BLOCKER"
    assert by_epc["EPC2"]["sentinel_scene_date"] == ""
    # Confirmed same patch: date IS filled (linkable).
    assert by_epc["EPC3"]["sentinel_date_status"] == "SENTINEL_DATE_CONFIRMED_SAME_PATCH"
    assert by_epc["EPC3"]["sentinel_scene_date"] == "2022-05-25"
    assert all(r["sentinel_date_inferred"] == "false" for r in rows)
