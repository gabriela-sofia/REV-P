import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_review_packet_pending_only(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    common.run_event_geometry_candidate_builder(common.parse_args([]))
    common.run_crosswalk_geometry_join_builder(common.parse_args([]))
    rows = common.run_patch_link_review_packet_builder(common.parse_args([]))
    assert rows
    assert all(r["review_status"] == "PENDING_PATCH_LINK_REVIEW" for r in rows)
    assert all("GROUND_TRUTH_VALIDATED" in r["forbidden_decisions"] for r in rows)
    # only C3/C4 candidates get packets (7 strong in the synthetic stack)
    assert len(rows) == 7
