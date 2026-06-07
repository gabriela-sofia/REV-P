import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all, write_csv


def test_rank1_collect_geometry_when_missing(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    write_csv(protocol / "v2ap_patch_event_link_readiness.csv",
              ["candidate_id", "patch_level_reference_candidate", "has_explicit_sentinel_crosswalk"],
              [{"candidate_id": "X", "patch_level_reference_candidate": "false",
                "has_explicit_sentinel_crosswalk": "false"}])
    write_csv(protocol / "v2ap_spatial_geometry_readiness.csv",
              ["candidate_id", "has_event_geometry"], [{"candidate_id": "X", "has_event_geometry": "false"}])
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "COLLECT_EVENT_GEOMETRY_FOR_TOP_CANDIDATES"


def test_rank1_resolve_crosswalk_when_geometry_ok(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    write_csv(protocol / "v2ap_patch_event_link_readiness.csv",
              ["candidate_id", "patch_level_reference_candidate", "has_explicit_sentinel_crosswalk"],
              [{"candidate_id": "X", "patch_level_reference_candidate": "false",
                "has_explicit_sentinel_crosswalk": "false"}])
    write_csv(protocol / "v2ap_spatial_geometry_readiness.csv",
              ["candidate_id", "has_event_geometry"], [{"candidate_id": "X", "has_event_geometry": "true"}])
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "RESOLVE_SENTINEL_PATCH_DATE_CROSSWALK"


def test_no_forbidden_action_allowed(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    write_csv(protocol / "v2ap_patch_event_link_readiness.csv",
              ["candidate_id", "patch_level_reference_candidate", "has_explicit_sentinel_crosswalk"],
              [{"candidate_id": "X", "patch_level_reference_candidate": "false",
                "has_explicit_sentinel_crosswalk": "false"}])
    write_csv(protocol / "v2ap_spatial_geometry_readiness.csv",
              ["candidate_id", "has_event_geometry"], [{"candidate_id": "X", "has_event_geometry": "false"}])
    rows = common.run_next_action_ranker(common.parse_args([]))
    for r in rows:
        if r["allowed"] == "true":
            name = r["next_action"].lower()
            assert "training" not in name and "protocol_b" not in name
            assert "overlay" not in name and "label" not in name
