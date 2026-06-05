import scripts.protocolo_c.revp_v2ah_common as common
from tests.test_revp_v2ah_common import install_inputs, set_env


def test_safe_tcc_export_contains_allowed_and_forbidden_claims(tmp_path, monkeypatch):
    data = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    common.run_ground_truth_search_stop_gate(common.parse_args([]))
    common.run_candidate_reference_review_queue(common.parse_args([]))
    rows = common.run_safe_tcc_export_builder(common.parse_args([]))
    assert rows
    assert all(r["claim_allowed"] for r in rows)
    assert all(r["claim_forbidden"] for r in rows)
    assert any("ground truth validado" in r["unsafe_wording"] for r in rows)
    assert all(r["forbidden_use"] for r in rows)
