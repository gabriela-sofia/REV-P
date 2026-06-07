import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_ground_truth_blocked_for_all(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_ground_truth_blocker_audit(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["ground_truth_blocked"] == "true" for r in rows)
    assert all(r["missing_explicit_sentinel_crosswalk"] == "true" for r in rows)
    assert all(r["missing_human_review"] == "true" for r in rows)
