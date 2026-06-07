import scripts.protocolo_c.revp_v2an_common as common
from tests.test_revp_v2an_common import install_all


def test_nine_dossiers_created(tmp_path, monkeypatch):
    data, protocol, docs, dossiers = install_all(tmp_path, monkeypatch)
    common.run_gate_closure_matrix_builder(common.parse_args([]))
    common.run_ground_reference_readiness_scorer(common.parse_args([]))
    index = common.run_candidate_dossier_builder(common.parse_args([]))
    assert len(index) == 9
    files = list(dossiers.glob("v2an_dossier_*.md"))
    assert len(files) == 9
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert "NOT_ESTABLISHED" in text
        common.assert_safe_text(text)
    assert all(r["ground_truth_status"] == "NOT_ESTABLISHED" for r in index)
