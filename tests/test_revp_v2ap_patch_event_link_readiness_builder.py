import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all


def _prep(common_mod):
    common_mod.run_patch_registry_inventory_builder(common_mod.parse_args([]))
    common_mod.run_spatial_geometry_readiness_builder(common_mod.parse_args([]))
    common_mod.run_sentinel_asset_inventory_builder(common_mod.parse_args([]))
    common_mod.run_temporal_window_builder(common_mod.parse_args([]))
    common_mod.run_sentinel_crosswalk_candidate_builder(common_mod.parse_args([]))


def test_patch_truth_always_false(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    _prep(common)
    rows = common.run_patch_event_link_readiness_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["patch_truth_allowed"] == "false" for r in rows)
    # patch_level_reference_candidate requires geometry + patch + explicit crosswalk
    for r in rows:
        if r["patch_level_reference_candidate"] == "true":
            assert r["has_event_geometry"] == "true"
            assert r["has_explicit_sentinel_crosswalk"] == "true"
