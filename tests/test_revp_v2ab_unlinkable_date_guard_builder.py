from tests.test_revp_v2ab_patch_namespace_inventory import install_base_inputs, set_env
import scripts.protocolo_c.revp_v2ab_common as common


def test_unlinkable_guard_blocks_refpatch_to_numeric_without_crosswalk(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan)
    common.run_patch_namespace_inventory(common.parse_args([]))
    common.run_patch_identity_crosswalk_audit(common.parse_args([]))
    rows = common.run_unlinkable_date_guard_builder(common.parse_args([]))
    date_patches = {r["date_patch_id"] for r in rows}
    # REFPATCH and scaffolding recovered dates are guarded as unlinkable.
    assert "REFPATCH_REC_001" in date_patches
    assert "REC_PATCH_A" in date_patches
    for r in rows:
        assert r["event_patch_namespace"] == common.NS_EVENT
        assert "apply_date_by_region" in r["prohibited_use"]
        assert "apply_date_by_name_similarity" in r["prohibited_use"]
        assert r["sentinel_date_inferred"] == "false"
        assert r["can_create_ground_reference"] == "false"


def test_unlinkable_guard_skips_pairs_with_explicit_crosswalk(tmp_path, monkeypatch):
    data, scan = set_env(tmp_path, monkeypatch)
    install_base_inputs(data, scan, with_explicit_crosswalk=True)
    common.run_patch_namespace_inventory(common.parse_args([]))
    common.run_patch_identity_crosswalk_audit(common.parse_args([]))
    rows = common.run_unlinkable_date_guard_builder(common.parse_args([]))
    # With an explicit EVENT<->ANCHOR crosswalk, REFPATCH dates are no longer
    # guarded as unlinkable (they become linkable through the explicit key).
    assert "REFPATCH_REC_001" not in {r["date_patch_id"] for r in rows}
