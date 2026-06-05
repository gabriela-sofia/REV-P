import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, set_env


def test_next_action_ranker_keeps_operational_actions_blocked(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "HUMAN_REVIEW_EXECUTION_OR_SAFE_TCC_EXPORT"
    assert rows[-1]["allowed"] == "false"
    assert "TRAINING" in rows[-1]["next_action"]
