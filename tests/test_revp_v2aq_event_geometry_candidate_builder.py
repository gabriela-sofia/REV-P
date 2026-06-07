import scripts.protocolo_c.revp_v2aq_common as common
from tests.test_revp_v2aq_common import install_all


def test_event_geometry_no_invention(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson = install_all(tmp_path, monkeypatch)
    rows = common.run_event_geometry_candidate_builder(common.parse_args([]))
    assert len(rows) == 9
    assert all(r["can_use_for_ground_truth"] == "false" for r in rows)
    assert all(r["geometry_status"] in common.ALLOWED_GEOMETRY_STATUS for r in rows)
    statuses = {r["geometry_status"] for r in rows}
    assert "OFFICIAL_MAP_DIGITIZATION_REQUIRED" in statuses
    assert "INSUFFICIENT_GEOMETRY" in statuses
    # patch-link usable only for map/point digitizable candidates
    by = {r["candidate_id"]: r for r in rows}
    assert by["PET_2022_02_15"]["can_use_for_patch_link_review"] == "true"
    assert by["PET_2024_03_21_28"]["can_use_for_patch_link_review"] == "false"
