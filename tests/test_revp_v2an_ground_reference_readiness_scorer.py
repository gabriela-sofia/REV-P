import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_scorer_blocks_ground_truth_and_label(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_ground_reference_readiness_scorer(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["can_create_ground_truth"] == "false" for r in rows)
    assert all(r["can_create_label"] == "false" for r in rows)
    assert all(r["protocol_b_status"] == "BLOCKED" for r in rows)
    # strong candidates separated from blocked ones
    bands = {r["readiness_band"] for r in rows}
    assert bands  # at least populated
    assert any(r["can_enter_human_ground_reference_review"] == "true" for r in rows)
