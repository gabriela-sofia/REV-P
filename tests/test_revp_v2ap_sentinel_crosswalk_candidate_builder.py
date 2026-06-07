import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all


def test_crosswalk_candidates_explicit_only(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    common.run_sentinel_asset_inventory_builder(common.parse_args([]))
    common.run_temporal_window_builder(common.parse_args([]))
    rows = common.run_sentinel_crosswalk_candidate_builder(common.parse_args([]))
    assert rows
    for r in rows:
        if r["can_be_used_as_explicit_crosswalk"] == "true":
            assert r["crosswalk_evidence_type"] in ("manifest_field_explicit", "filename_date_explicit", "metadata_field_explicit")
            assert r["region_match"] == "true"
            assert r["asset_date"]
    # the synthetic Recife asset (2022-05-26) should match REC_2022_05_24_30 window
    rec = [r for r in rows if r["candidate_id"] == "REC_2022_05_24_30"]
    assert any(r["within_temporal_window"] == "true" for r in rec)


def test_no_dino_crosswalk(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    common.run_sentinel_asset_inventory_builder(common.parse_args([]))
    common.run_temporal_window_builder(common.parse_args([]))
    rows = common.run_sentinel_crosswalk_candidate_builder(common.parse_args([]))
    # no explicit crosswalk may rest on a dino source
    assert all(r["crosswalk_evidence_type"] not in ("dino_similarity", "visual_similarity") for r in rows)
