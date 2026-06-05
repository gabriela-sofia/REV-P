import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, set_env


def test_review_outcome_defaults_pending_and_blocked(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    rows = common.run_review_outcome_registry_builder(common.parse_args([]))
    assert all(r["review_outcome_status"] == "PENDING_HUMAN_REVIEW" for r in rows)
    assert all(r["human_review_completed"] == "false" for r in rows)
    assert all(r["adjudication_completed"] == "false" for r in rows)
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
