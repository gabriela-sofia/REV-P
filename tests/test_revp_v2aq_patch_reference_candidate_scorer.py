import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_scorer_blocks_everything(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    common.run_event_geometry_candidate_builder(common.parse_args([]))
    common.run_crosswalk_geometry_join_builder(common.parse_args([]))
    rows = common.run_patch_reference_candidate_scorer(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["can_create_ground_truth"] == "false" for r in rows)
    assert all(r["can_create_label"] == "false" for r in rows)
    assert all(r["protocol_b_status"] == "BLOCKED" for r in rows)
    for r in rows:
        assert int(r["overall_patch_reference_score"]) == (
            int(r["event_geometry_score"]) + int(r["patch_geometry_score"]) +
            int(r["sentinel_crosswalk_score"]) + int(r["source_trace_score"]))
