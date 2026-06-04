import scripts.protocolo_c.revp_v1uo_multiregion_common as common
from tests.test_revp_v1uo_multiregion_event_registry_builder import make_base


def test_ranker_generates_next_action(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_multiregion_event_registry_builder(str(data / "v1uo_multiregion_event_registry.csv"))
    common.run_multiregion_candidate_router(str(data / "v1uo_multiregion_candidate_router.csv"))
    rows = common.run_ground_truth_opportunity_ranker(str(data / "rank.csv"))
    assert rows[0]["recommended_next_version"] == "v1up"
    assert rows[0]["recommended_next_action"]
    assert rows[0]["event_id"] == "PET_2024_03_21_28"
