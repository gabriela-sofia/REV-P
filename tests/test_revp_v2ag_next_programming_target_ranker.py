import scripts.protocolo_c.revp_v2ag_common as common
from tests.test_revp_v2ag_crosswalk_source_inventory import install_packages, set_env


def test_ranker_selects_by_score_when_no_strong_crosswalk(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_packages(data)
    common.run_crosswalk_source_inventory(common.parse_args([]))
    common.run_explicit_crosswalk_detector(common.parse_args([]))
    common.run_lineage_crosswalk_candidate_builder(common.parse_args([]))
    common.run_sentinel_date_linkability_auditor(common.parse_args([]))
    rows = common.run_next_programming_target_ranker(common.parse_args([]))
    scores = [int(r["score"]) for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert rows[0]["next_target"] == "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE"
