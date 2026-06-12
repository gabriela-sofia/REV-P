import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all


def _prep():
    common.run_deep_probe_priority_builder(common.parse_args([]))
    common.run_geometry_payload_detector(common.parse_args([]))
    common.run_coordinate_sanity_validator(common.parse_args([]))
    common.run_geometry_candidate_classifier(common.parse_args([]))
    common.run_geojson_candidate_exporter(common.parse_args([]))
    common.run_patch_link_readiness_update(common.parse_args([]))


def test_gap_report_offline(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    _prep()
    rows = common.run_digitization_gap_report_builder(common.parse_args([]))
    assert rows
    assert all(r["do_not_infer"] == "true" for r in rows)
    assert all(r["missing_geometry"] == "true" for r in rows)
    assert all(r["missing_license"] == "true" for r in rows)
    by = {r["candidate_id"]: r for r in rows}
    assert by["PET_2022_02_15"]["recommended_action"] == "MANUAL_DIGITIZE_FROM_OFFICIAL_SOURCE"
