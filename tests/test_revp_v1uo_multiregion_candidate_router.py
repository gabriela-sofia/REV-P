import scripts.protocolo_c.revp_v1uo_multiregion_common as common
from tests.test_revp_v1uo_multiregion_event_registry_builder import make_base


def test_candidate_router_does_not_create_ground_reference(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_multiregion_event_registry_builder(str(data / "v1uo_multiregion_event_registry.csv"))
    rows = common.run_multiregion_candidate_router(str(data / "router.csv"))
    by_event = {r["event_id"]: r for r in rows}
    assert by_event["REC_2022_05_24_30"]["candidate_class"] == "locality-only candidate"
    assert by_event["REC_2022_05_24_30"]["has_geometry_support"] == "false"
    assert all(r["can_advance_to_ground_reference"] == "false" for r in rows)
    assert all(r["can_create_training_label"] == "false" for r in rows)
