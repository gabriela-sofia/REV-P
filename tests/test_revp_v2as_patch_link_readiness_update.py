import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all, inject_explicit_geometry


def _prep():
    common.run_deep_probe_priority_builder(common.parse_args([]))
    common.run_geometry_payload_detector(common.parse_args([]))
    common.run_coordinate_sanity_validator(common.parse_args([]))
    common.run_geometry_candidate_classifier(common.parse_args([]))
    common.run_geojson_candidate_exporter(common.parse_args([]))


def test_not_ready_offline(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    _prep()
    rows = common.run_patch_link_readiness_update(common.parse_args([]))
    assert all(r["patch_link_review_ready"] == "false" for r in rows)
    assert all(r["patch_truth_allowed"] == "false" for r in rows)


def test_ready_requires_geometry_patch_and_crosswalk(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    # REC_2023_02_05_06 has crosswalk + patch in the fixture; add explicit geometry
    inject_explicit_geometry(protocol, "REC_2023_02_05_06", {"type": "Point", "coordinates": [-34.9, -8.05]})
    _prep()
    rows = common.run_patch_link_readiness_update(common.parse_args([]))
    by = {r["candidate_id"]: r for r in rows}
    assert by["REC_2023_02_05_06"]["patch_link_review_ready"] == "true"
    assert by["REC_2023_02_05_06"]["patch_truth_allowed"] == "false"
    # PET has no patch/crosswalk -> stays not ready even with geometry absent
    assert by["PET_2022_02_15"]["patch_link_review_ready"] == "false"
