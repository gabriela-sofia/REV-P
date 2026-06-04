import scripts.protocolo_c.revp_v1un_recife_common as common
from tests.test_revp_v1un_recife_human_review_evidence_consolidator import make_base


def test_limitations_matrix_contains_no_coordinates_and_no_overlay(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    rows = common.run_limitations_matrix_builder(str(data / "limitations.csv"))
    limitations = {r["limitation"] for r in rows}
    assert "no_coordinates" in limitations
    assert "no_overlay" in limitations
    assert all(r["ground_truth_operational"] == "false" for r in rows)
