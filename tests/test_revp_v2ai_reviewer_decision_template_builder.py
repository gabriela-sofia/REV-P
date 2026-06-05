import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, set_env


def test_reviewer_decision_template_stays_pending(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    common.run_review_assignment_builder(common.parse_args([]))
    rows = common.run_reviewer_decision_template_builder(common.parse_args([]))
    assert len(rows) == 4
    assert all(r["decision_status"] == "PENDING_HUMAN_REVIEW" for r in rows)
    assert all(r["phenomenon_observed"] == "UNREVIEWED" for r in rows)
    assert all(r["needs_adjudication"] == "UNKNOWN_UNTIL_REVIEW" for r in rows)
    assert all(r["decision_timestamp"] == "" for r in rows)
