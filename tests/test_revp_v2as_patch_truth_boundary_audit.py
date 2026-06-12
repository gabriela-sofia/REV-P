import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all


def _prep():
    common.run_deep_probe_priority_builder(common.parse_args([]))
    common.run_geometry_payload_detector(common.parse_args([]))
    common.run_coordinate_sanity_validator(common.parse_args([]))
    common.run_geometry_candidate_classifier(common.parse_args([]))
    common.run_geojson_candidate_exporter(common.parse_args([]))
    common.run_geojson_validation(common.parse_args([]))
    common.run_patch_link_readiness_update(common.parse_args([]))


def test_boundary_keeps_everything_blocked(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    _prep()
    rows = common.run_patch_truth_boundary_audit(common.parse_args([]))
    assert rows
    assert all(r["patch_truth_allowed"] == "false" for r in rows)
    assert all(r["can_create_ground_truth"] == "false" for r in rows)
    assert all(r["can_create_label"] == "false" for r in rows)
    assert all(r["protocol_b_status"] == "BLOCKED" for r in rows)
    assert all(r["external_validation_pending"] == "true" for r in rows)
