import scripts.protocolo_c.revp_v2aj_common as common
from tests.test_revp_v2aj_common import install_inputs, set_env


def test_claims_matrix_separates_allowed_and_forbidden(tmp_path, monkeypatch):
    data, _ = set_env(tmp_path, monkeypatch)
    install_inputs(data)
    rows = common.run_tcc_claims_matrix_builder(common.parse_args([]))
    assert sum(1 for r in rows if r["claim_allowed"] == "true") >= 8
    assert any(r["claim_allowed"] == "false" for r in rows)
    assert any("camada revisavel" in r["safe_wording"] for r in rows)
    assert any("ground truth validado" in r["unsafe_wording"] for r in rows)
