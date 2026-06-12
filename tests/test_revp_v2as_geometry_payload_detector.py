import scripts.protocolo_c.revp_v2as_common as common
from tests.test_revp_v2as_common import install_all, inject_explicit_geometry


def test_no_payload_offline(tmp_path, monkeypatch):
    install_all(tmp_path, monkeypatch)
    common.run_deep_probe_priority_builder(common.parse_args([]))
    rows = common.run_geometry_payload_detector(common.parse_args([]))
    assert all(r["explicit_geometry_found"] == "false" for r in rows)
    assert all(r["geometry_detection_status"] == "NO_EXPLICIT_GEOMETRY_PAYLOAD" for r in rows)


def test_detects_explicit_payload(tmp_path, monkeypatch):
    datasets, protocol, docs, geojson, cache = install_all(tmp_path, monkeypatch)
    inject_explicit_geometry(protocol, "REC_2023_02_05_06", {"type": "Point", "coordinates": [-34.9, -8.05]})
    common.run_deep_probe_priority_builder(common.parse_args([]))
    rows = common.run_geometry_payload_detector(common.parse_args([]))
    by = {r["candidate_id"]: r for r in rows}
    assert by["REC_2023_02_05_06"]["explicit_geometry_found"] == "true"
    assert by["REC_2023_02_05_06"]["geometry_detection_status"] == "EXPLICIT_GEOMETRY_PAYLOAD_DETECTED"
