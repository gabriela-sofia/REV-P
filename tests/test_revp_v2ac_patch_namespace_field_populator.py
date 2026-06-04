from tests.test_revp_v2ac_event_patch_v2_package_builder import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ac_common as common


def test_namespace_populator_identifies_event_patch_candidate_namespace(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_base_inputs(data)
    common.run_event_patch_v2_package_builder(common.parse_args([]))
    rows = common.run_patch_namespace_field_populator(common.parse_args([]))
    by_epc = {r["event_patch_candidate_id"]: r for r in rows}
    assert by_epc["EPC0"]["patch_namespace"] == common.NS_EVENT
    assert by_epc["EPC0"]["namespace_population_status"] == "NAMESPACE_POPULATED"
    # Empty patch id -> PATCH_ID_MISSING namespace, never anchor.
    assert by_epc["EPC2"]["patch_namespace"] == "PATCH_ID_MISSING"
    assert by_epc["EPC2"]["namespace_population_status"] == "PATCH_ID_MISSING"
    assert all("ANCHOR" not in r["patch_namespace"] for r in rows)
