import scripts.protocolo_c.revp_v2ai_common as common
from tests.test_revp_v2ai_common import install_v2ah, set_env


def test_safe_promotion_blockers_always_block_promotion(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_v2ah(data)
    rows = common.run_safe_promotion_blockers(common.parse_args([]))
    assert len(rows) == 2
    assert all(r["promotion_allowed"] == "false" for r in rows)
    assert all(r["blocker_human_review"] == "true" for r in rows)
    assert all(r["blocker_adjudication"] == "true" for r in rows)
