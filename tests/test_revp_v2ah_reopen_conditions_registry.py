import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import set_env


def test_reopen_conditions_require_review(tmp_path, monkeypatch):
    set_env(tmp_path, monkeypatch)
    rows = common.run_reopen_conditions_registry(common.parse_args([]))
    assert len(rows) == 7
    assert all(r["can_reopen_if_met"] == "review_gate_only" for r in rows)
    assert all(r["still_forbidden_without_human_review"] == "true" for r in rows)
