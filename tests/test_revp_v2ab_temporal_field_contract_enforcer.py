from tests.test_revp_v2ab_patch_namespace_inventory import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ab_common as common


def test_temporal_contract_never_applies_unlinkable_date(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    common.run_patch_namespace_inventory(common.parse_args([]))
    rows = common.run_temporal_field_contract_enforcer(common.parse_args([]))
    by_epc = {r["event_patch_candidate_id"]: r for r in rows}
    # REC_00001: a parallel-namespace date exists for REC but is unlinkable.
    rec = by_epc["EPC0"]
    assert rec["sentinel_date_status"] == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE"
    assert rec["selected_sentinel_date"] == ""  # unlinkable date is NEVER applied
    assert rec["date_source_namespace"] in {common.NS_ANCHOR, common.NS_SCAFFOLD}
    # CUR candidate with empty patch -> missing with blocker.
    cur = by_epc["EPC2"]
    assert cur["sentinel_date_status"] == "SENTINEL_DATE_MISSING_WITH_BLOCKER"
    # REC_00009: its OWN patch recovered a usable date -> confirmed same patch.
    same = by_epc["EPC3"]
    assert same["sentinel_date_status"] == "SENTINEL_DATE_CONFIRMED_SAME_PATCH"
    assert same["selected_sentinel_date"] == "2022-05-25"
    assert all(r["sentinel_date_inferred"] == "false" for r in rows)
