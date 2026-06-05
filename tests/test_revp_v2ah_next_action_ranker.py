import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import install_inputs, set_env


def test_next_action_ranker_does_not_recommend_operational_modeling(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_ground_truth_search_stop_gate(common.parse_args([]))
    common.run_candidate_reference_review_queue(common.parse_args([]))
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] in {"HUMAN_REVIEW_ADJUDICATION_PACKAGE", "SAFE_TCC_EVIDENCE_EXPORT"}
    assert rows[-1]["next_action"] == "OPERATIONAL_MODELING_TRAINING_OVERLAY"
    assert rows[-1]["allowed"] == "false"
