import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, set_env


def test_adjudication_queue_waits_for_human_review(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    common.run_review_assignment_builder(common.parse_args([]))
    rows = common.run_adjudication_queue_builder(common.parse_args([]))
    assert len(rows) == 2
    assert all(r["adjudication_status"] == "WAITING_FOR_HUMAN_REVIEW" for r in rows)
    assert all(r["can_promote_after_adjudication"] == "false" for r in rows)
    assert all(r["still_requires_external_evidence"] == "true" for r in rows)
