import scripts.protocolo_c.revp_v1um_recife_common as common
from tests.test_revp_v1um_recife_locality_candidate_sampler import make_base


def test_hazard_strong_increases_priority_and_generic_stays_review_only(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    rows = common.run_hazard_semantics_ranker(str(data / "hazard.csv"))
    by_id = {r["candidate_row_id"]: r for r in rows}
    assert by_id["c1"]["hazard_class"] == "FLOOD_STRONG"
    assert by_id["c1"]["review_priority_delta"] == "-2"
    assert by_id["c3"]["hazard_class"] == "CIVIL_DEFENSE_GENERIC"
    assert by_id["c3"]["can_create_training_label"] == "false"
