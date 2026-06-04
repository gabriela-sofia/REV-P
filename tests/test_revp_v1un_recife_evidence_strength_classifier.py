import scripts.protocolo_c.revp_v1un_recife_common as common
from tests.test_revp_v1un_recife_human_review_evidence_consolidator import make_base


def test_evidence_strength_blocks_coordinate_overlay_and_ground_reference(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    rows = common.run_evidence_strength_classifier(str(data / "strength.csv"))
    by_dim = {r["evidence_dimension"]: r["classification"] for r in rows}
    assert by_dim["coordinate_support"] == "BLOCKED"
    assert by_dim["overlay_support"] == "BLOCKED"
    assert by_dim["ground_reference_support"] == "BLOCKED"
    assert all(r["can_create_training_label"] == "false" for r in rows)
