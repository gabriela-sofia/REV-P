import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all, inject_explicit_geometry


def _prep():
    common.run_deep_probe_priority_builder(common.parse_args([]))
    common.run_geometry_payload_detector(common.parse_args([]))
    common.run_coordinate_sanity_validator(common.parse_args([]))


def test_offline_classification_no_real_geometry(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    _prep()
    rows = common.run_geometry_candidate_classifier(common.parse_args([]))
    assert all(r["can_export_real_geometry"] == "false" for r in rows)
    assert all(r["can_use_for_ground_truth"] == "false" for r in rows)
    by = {r["candidate_id"]: r for r in rows}
    assert by["PET_2022_02_15"]["geometry_status"] == "MANUAL_DIGITIZATION_REQUIRED"
    assert by["PET_2024_03_21_28"]["geometry_status"] == "NO_EXPLICIT_GEOMETRY_STILL_NULL"


def test_explicit_coordinate_classified_exportable(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    inject_explicit_geometry(protocol, "REC_2023_02_05_06", {"type": "Point", "coordinates": [-34.9, -8.05]})
    _prep()
    rows = common.run_geometry_candidate_classifier(common.parse_args([]))
    by = {r["candidate_id"]: r for r in rows}
    assert by["REC_2023_02_05_06"]["geometry_status"] == "EXPLICIT_COORDINATE_AVAILABLE"
    assert by["REC_2023_02_05_06"]["can_export_real_geometry"] == "true"
    assert by["REC_2023_02_05_06"]["can_use_for_ground_truth"] == "false"
