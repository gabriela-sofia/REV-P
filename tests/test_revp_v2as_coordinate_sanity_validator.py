import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all, inject_explicit_geometry


def _prep():
    common.run_deep_probe_priority_builder(common.parse_args([]))


def test_no_explicit_coordinate_offline(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    _prep()
    rows = common.run_coordinate_sanity_validator(common.parse_args([]))
    assert all(r["coordinate_validation_status"] == "NO_EXPLICIT_COORDINATE" for r in rows)


def test_coordinate_inside_brazil(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    inject_explicit_geometry(protocol, "REC_2023_02_05_06", {"type": "Point", "coordinates": [-34.9, -8.05]})
    _prep()
    rows = common.run_coordinate_sanity_validator(common.parse_args([]))
    by = {r["candidate_id"]: r for r in rows}
    assert by["REC_2023_02_05_06"]["inside_brazil_bounds"] == "true"
    assert by["REC_2023_02_05_06"]["coordinate_validation_status"] == "EXPLICIT_COORDINATE_PLAUSIBLE"


def test_coordinate_outside_brazil(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    inject_explicit_geometry(protocol, "REC_2023_02_05_06", {"type": "Point", "coordinates": [10.0, 50.0]})
    _prep()
    rows = common.run_coordinate_sanity_validator(common.parse_args([]))
    by = {r["candidate_id"]: r for r in rows}
    assert by["REC_2023_02_05_06"]["inside_brazil_bounds"] == "false"
    assert by["REC_2023_02_05_06"]["coordinate_validation_status"] == "OUTSIDE_BRAZIL_BOUNDS"
