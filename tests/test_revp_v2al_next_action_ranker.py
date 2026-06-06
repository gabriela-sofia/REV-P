import scripts.protocolo_c.revp_v2al_common as common
from tests.test_revp_v2al_common import install_all, read_csv


def test_next_action_rank_1_is_manual_integration(tmp_path, monkeypatch):
    data, _, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_next_action_ranker(common.parse_args([]))
    assert rows[0]["next_action"] == "MANUAL_TCC_INTEGRATION_REVIEW"
    assert rows[0]["rank"] == "1"
    actions = {r["next_action"] for r in rows}
    assert "ORIENTATION_MEETING_REVIEW" in actions
    assert "COPY_SECTIONS_TO_MANUSCRIPT_AFTER_HUMAN_REVIEW" in actions
    assert "APPENDIX_EXPORT" in actions
    assert "HUMAN_REVIEW_EXECUTION" in actions
    blocked = [r for r in rows
               if r["next_action"] == "TRAINING_PROTOCOL_B_OVERLAY_LABEL_GT_PROMOTION"][0]
    assert blocked["allowed"] == "false"


def test_no_forbidden_action_is_allowed(tmp_path, monkeypatch):
    data, _, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_next_action_ranker(common.parse_args([]))
    for r in rows:
        if r["allowed"] == "true":
            name = r["next_action"].lower()
            assert "training" not in name
            assert "protocol_b" not in name
            assert "overlay" not in name
            assert "_gt_" not in name and "ground_truth" not in name
