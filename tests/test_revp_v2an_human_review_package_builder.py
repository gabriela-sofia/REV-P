import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_review_package_pending_only(tmp_path, monkeypatch):
    data, protocol, docs, _ = install_all(tmp_path, monkeypatch)
    common.run_gate_closure_matrix_builder(common.parse_args([]))
    common.run_ground_reference_readiness_scorer(common.parse_args([]))
    rows = common.run_human_review_package_builder(common.parse_args([]))
    assert rows
    assert all(r["review_status"] == "PENDING_HUMAN_REVIEW" for r in rows)
    assert all("ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW" in r["possible_decisions"] for r in rows)
    assert all("GROUND_TRUTH_VALIDATED" in r["forbidden_decisions"] for r in rows)
