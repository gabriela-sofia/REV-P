import scripts.protocolo_c.revp_v1uo_multiregion_common as common
from tests.test_revp_v1uo_multiregion_event_registry_builder import make_base


def test_event_patch_package_does_not_invent_patch_id(tmp_path, monkeypatch):
    data = make_base(tmp_path, monkeypatch)
    common.run_multiregion_event_registry_builder(str(data / "v1uo_multiregion_event_registry.csv"))
    common.run_multiregion_candidate_router(str(data / "v1uo_multiregion_candidate_router.csv"))
    rows = common.run_event_patch_package_prebuilder(str(data / "packages.csv"))
    assert rows[0]["patch_id"] == "PATCH_LINKAGE_MISSING"
    assert rows[0]["sentinel_scene_date"] == "PATCH_LINKAGE_MISSING"
    assert all(r["can_create_ground_reference"] == "false" for r in rows)
