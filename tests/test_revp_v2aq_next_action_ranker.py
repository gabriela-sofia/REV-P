import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all, write_csv


def _write_inputs(protocol, review_ready, geometry_present):
    write_csv(protocol / "v2aq_crosswalk_geometry_join_candidates.csv",
              ["candidate_id", "patch_level_review_ready", "has_crosswalk_candidate"],
              [{"candidate_id": "X", "patch_level_review_ready": review_ready,
                "has_crosswalk_candidate": "true"}])
    write_csv(protocol / "v2aq_geojson_candidate_index.csv",
              ["candidate_id", "geometry_present"],
              [{"candidate_id": "X", "geometry_present": geometry_present}])


def test_rank1_execute_when_review_ready(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    _write_inputs(protocol, "true", "true")
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "EXECUTE_PATCH_LINK_REVIEW_WITH_EXPLICIT_GEOMETRY"


def test_rank1_digitize_when_null_geojson(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    _write_inputs(protocol, "false", "false")
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "DIGITIZE_EVENT_GEOMETRY_FOR_TOP_CANDIDATES"


def test_no_forbidden_action_allowed(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    _write_inputs(protocol, "false", "false")
    rows = common.run_next_action_ranker(common.parse_args([]))
    for r in rows:
        if r["allowed"] == "true":
            name = r["next_action"].lower()
            assert "training" not in name and "protocol_b" not in name
            assert "overlay" not in name and "label" not in name
