import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env


def test_next_action_ranker_prefers_safe_tcc_writeup(tmp_path, monkeypatch):
    data, _ = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "SAFE_TCC_PROTOCOL_C_WRITEUP"
    assert rows[-1]["allowed"] == "false"
    assert "TRAINING" in rows[-1]["next_action"]
