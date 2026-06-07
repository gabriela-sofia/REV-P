import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_g9_always_blocked(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    rows = common.run_gate_closure_matrix_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["g9_promotion_decision"] == "BLOCKED_PENDING_HUMAN_REVIEW" for r in rows)
    for r in rows:
        assert int(r["closed_gates_count"]) >= 0


def test_open_spatial_gate_when_g4_open(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch, "candidate_no_spatial.csv")
    # candidate_no_spatial has only 1 row; loader requires 9, so build via derive path
    import pytest
    with pytest.raises(ValueError):
        common.run_gate_closure_matrix_builder(common.parse_args([]))
