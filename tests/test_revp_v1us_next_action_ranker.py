import scripts.protocolo_c.revp_v1us_common as common
from tests.test_revp_v1us_patch_registry_resolver import set_env, build_chain


def test_ranker_picks_actions_from_real_blockers(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_next_action_ranker()
    by_event = {r["event_id"]: r for r in rows}
    assert by_event["REC_2022_05_24_30"]["next_action"] == "RECIFE_COORDINATE_RECOVERY"
    assert by_event["CUR_EVENT_REGISTRY_MISSING"]["next_action"] == "CURITIBA_EVENT_REGISTRY_DISCOVERY"
    assert by_event["PET_2022_02_15"]["next_action"] == "PETROPOLIS_GEOMETRY_SEARCH_EXHAUSTED_SWITCH_REGION"
    assert by_event["PET_2024_03_21_28"]["next_action"] == "PETROPOLIS_GEOMETRY_SEARCH_EXHAUSTED_SWITCH_REGION"


def test_ranker_top_is_recife_coordinate_recovery(tmp_path, monkeypatch):
    data, root, _, _ = set_env(tmp_path, monkeypatch)
    build_chain(data, root)
    rows = common.run_next_action_ranker()
    top = next(r for r in rows if r["rank"] == "1")
    assert top["next_action"] == "RECIFE_COORDINATE_RECOVERY"
