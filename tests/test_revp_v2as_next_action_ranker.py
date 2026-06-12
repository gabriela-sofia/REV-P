import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all, write_csv


def _write_inputs(protocol, geometry_present, review_ready="false"):
    write_csv(protocol / "v2as_geojson_candidate_index.csv",
              ["candidate_id", "geometry_present"],
              [{"candidate_id": "X", "geometry_present": geometry_present}])
    write_csv(protocol / "v2as_patch_link_readiness_update.csv",
              ["candidate_id", "patch_link_review_ready"],
              [{"candidate_id": "X", "patch_link_review_ready": review_ready}])


def test_rank1_manual_when_all_null(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    _write_inputs(protocol, "false")
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "MANUAL_DIGITIZE_EVENT_GEOMETRY_FROM_OFFICIAL_SOURCES"


def test_rank1_external_validate_when_real_geometry(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    _write_inputs(protocol, "true")
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "EXTERNAL_VALIDATE_EXPLICIT_EVENT_GEOMETRY"


def test_rank1_prepare_patch_link_when_review_ready(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    _write_inputs(protocol, "true", review_ready="true")
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "PREPARE_PATCH_LINK_REVIEW"


def test_no_forbidden_action_allowed(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    _write_inputs(protocol, "false")
    rows = common.run_next_action_ranker(common.parse_args([]))
    for r in rows:
        if r["allowed"] == "true":
            name = r["next_action"].lower()
            assert "training" not in name and "protocol_b" not in name
            assert "overlay" not in name and "label" not in name
