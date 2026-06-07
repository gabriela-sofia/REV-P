import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_match_no_overlay(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    common.run_event_geometry_candidate_builder(common.parse_args([]))
    rows = common.run_patch_geometry_match_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["overlay_executed"] == "false" for r in rows)
    assert all(r["manual_review_required"] == "true" for r in rows)
