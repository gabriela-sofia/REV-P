import scripts.protocolo_c.revp_v1uo_multiregion_common as common
from tests.test_revp_v1uo_multiregion_event_registry_builder import make_base


def test_evidence_matrix_classifies_coordinate_geometry_overlay(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_multiregion_event_registry_builder(str(data / "v1uo_multiregion_event_registry.csv"))
    common.run_multiregion_candidate_router(str(data / "v1uo_multiregion_candidate_router.csv"))
    rows = common.run_multiregion_evidence_matrix_builder(str(data / "matrix.csv"))
    rec = [r for r in rows if r["event_id"] == "REC_2022_05_24_30"]
    by_dim = {r["dimension"]: r["classification"] for r in rec}
    assert by_dim["coordinate_support"] == "BLOCKED"
    assert by_dim["geometry_support"] == "BLOCKED"
    assert by_dim["overlay_readiness"] == "BLOCKED"
