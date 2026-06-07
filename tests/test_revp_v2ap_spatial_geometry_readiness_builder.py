import scripts.protocolo_c.revp_v2ap_common as common
from tests.test_revp_v2ap_common import install_all


def test_geometry_readiness_no_invented_coords(tmp_path, monkeypatch):
    datasets, protocol, docs = install_all(tmp_path, monkeypatch)
    common.run_patch_registry_inventory_builder(common.parse_args([]))
    rows = common.run_spatial_geometry_readiness_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["has_event_coordinates"] == "false" for r in rows)
    valid = {"EVENT_GEOMETRY_READY", "PATCH_GEOMETRY_READY", "EVENT_AND_PATCH_GEOMETRY_READY",
             "ANCHOR_ONLY_NEEDS_GEOMETRY", "INSUFFICIENT_SPATIAL_EVIDENCE"}
    assert all(r["geometry_readiness_status"] in valid for r in rows)
