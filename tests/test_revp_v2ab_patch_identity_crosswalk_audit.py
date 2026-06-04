from tests.test_revp_v2ab_patch_namespace_inventory import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ab_common as common


def test_crosswalk_audit_does_not_use_region_or_row_order(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)  # no explicit crosswalk fixture
    rows = common.run_patch_identity_crosswalk_audit(common.parse_args([]))
    by_pair = {(r["source_namespace"], r["target_namespace"]): r for r in rows}
    # Numeric event ids and DINO ids share the literal patch_id -> explicit.
    dino = by_pair[(common.NS_EVENT, common.NS_DINO)]
    assert dino["crosswalk_status"] == "EXPLICIT_CROSSWALK_FOUND"
    assert "patch_id" in dino["crosswalk_key_fields"]
    # REFPATCH shares region REC with numeric ids but has no shared key field,
    # so region must NOT produce a crosswalk.
    anchor = by_pair[(common.NS_EVENT, common.NS_ANCHOR)]
    assert anchor["crosswalk_status"] == "NO_EXPLICIT_CROSSWALK"
    assert anchor["matched_pairs"] == "0"
    assert all(r["crosswalk_inferred"] == "false" for r in rows)


def test_crosswalk_audit_detects_explicit_key(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan, with_explicit_crosswalk=True)
    rows = common.run_patch_identity_crosswalk_audit(common.parse_args([]))
    anchor = next(r for r in rows if r["source_namespace"] == common.NS_EVENT and r["target_namespace"] == common.NS_ANCHOR)
    # An explicit refpatch_id<->patch_id row makes the crosswalk explicit.
    assert anchor["crosswalk_status"] == "EXPLICIT_CROSSWALK_FOUND"
    assert "refpatch_id" in anchor["crosswalk_key_fields"]
    assert anchor["crosswalk_inferred"] == "false"
