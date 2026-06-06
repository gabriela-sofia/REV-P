import scripts.protocolo_c.revp_v2am_common as common
from tests.test_revp_v2am_common import install_all


def test_next_action_rank_1(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "MANUAL_APPENDIX_REVIEW_AND_ORIENTATION_MEETING"
    actions = {r["next_action"] for r in rows}
    assert "INTEGRATE_APPROVED_SECTIONS_IN_TCC" in actions
    assert "HUMAN_REVIEW_EXECUTION" in actions
    assert "WAIT_FOR_NEW_QUALIFIED_SOURCE" in actions
    assert "FINAL_TCC_CLAIM_AUDIT" in actions


def test_no_forbidden_action_allowed(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    rows = common.run_next_action_ranker(common.parse_args([]))
    for r in rows:
        if r["allowed"] == "true":
            name = r["next_action"].lower()
            assert "training" not in name and "protocol_b" not in name
            assert "overlay" not in name and "ground_truth" not in name
