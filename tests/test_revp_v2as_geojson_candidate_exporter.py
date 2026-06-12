import json

import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all, inject_explicit_geometry


def _prep():
    common.run_deep_probe_priority_builder(common.parse_args([]))
    common.run_geometry_payload_detector(common.parse_args([]))
    common.run_coordinate_sanity_validator(common.parse_args([]))
    common.run_geometry_candidate_classifier(common.parse_args([]))


def test_all_null_offline(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    _prep()
    index = common.run_geojson_candidate_exporter(common.parse_args([]))
    assert all(r["geometry_present"] == "false" for r in index)
    fc = json.load(open(geojson / "v2as_event_geometry_pet-2022-02-15.geojson", encoding="utf-8"))
    props = fc["features"][0]["properties"]
    assert fc["features"][0]["geometry"] is None
    assert props["not_ground_truth"] is True
    assert props["patch_truth_allowed"] is False
    assert props["raw_data_versioned"] is False


def test_real_geometry_from_explicit_payload(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    inject_explicit_geometry(protocol, "REC_2023_02_05_06", {"type": "Point", "coordinates": [-34.9, -8.05]})
    _prep()
    index = common.run_geojson_candidate_exporter(common.parse_args([]))
    by = {r["candidate_id"]: r for r in index}
    assert by["REC_2023_02_05_06"]["geometry_present"] == "true"
    fc = json.load(open(geojson / "v2as_event_geometry_rec-2023-02-05-06.geojson", encoding="utf-8"))
    assert fc["features"][0]["geometry"]["type"] == "Point"
