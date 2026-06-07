import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_join_patch_truth_false_and_review_ready_rule(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    common.run_event_geometry_candidate_builder(common.parse_args([]))
    rows = common.run_crosswalk_geometry_join_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["patch_truth_allowed"] == "false" for r in rows)
    for r in rows:
        if r["patch_level_review_ready"] == "true":
            assert r["has_event_geometry_candidate"] == "true"
            assert r["has_patch_geometry"] == "true"
            assert r["has_crosswalk_candidate"] == "true"
