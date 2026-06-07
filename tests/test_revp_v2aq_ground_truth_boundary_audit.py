import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_ground_truth_boundary_all_blocked(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    common.run_event_geometry_candidate_builder(common.parse_args([]))
    common.run_crosswalk_geometry_join_builder(common.parse_args([]))
    rows = common.run_ground_truth_boundary_audit(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["ground_truth_blocked"] == "true" for r in rows)
    assert all(r["label_blocked"] == "true" for r in rows)
    assert all(r["protocol_b_blocked"] == "true" for r in rows)
