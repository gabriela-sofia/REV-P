from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_08_metadata_replay as replay
from mv2_data_08_metadata_provider_contracts import MetadataQueryTarget


def _target() -> MetadataQueryTarget:
    return MetadataQueryTarget(
        patch_id="P1",
        asset_id="A1",
        sensor_family="SENTINEL_2",
        temporal_window_start="2022-01-01",
        temporal_window_end="2022-01-10",
        aoi_wgs84=[-35.0, -8.1, -34.9, -8.0],
    )


def test_hash_is_stable_and_order_independent() -> None:
    a = replay.hash_raw_response({"x": 1, "y": 2})
    b = replay.hash_raw_response({"y": 2, "x": 1})
    assert a == b
    assert len(a) == 64


def test_redact_drops_secrets_and_signed_urls() -> None:
    raw = {
        "token": "abc",
        "access_key": "secret",
        "local_path": "C:/Users/x/raster.tif",
        "href": "https://example.com/item?token=zzz",
        "safe": "keep",
        "nested": {"password": "p", "ok": 1},
    }
    clean = replay.redact_sensitive_fields(raw)
    assert "token" not in clean
    assert "access_key" not in clean
    assert "local_path" not in clean
    assert clean["safe"] == "keep"
    assert clean["nested"] == {"ok": 1}
    assert clean["href"] == "<redacted_signed_url>"


def test_empty_fixture_normalizes_to_no_items() -> None:
    assert replay.normalize_gee_response({"features": []}, _target()) == []
    assert replay.normalize_stac_response({"features": []}, _target()) == []
    assert replay.normalize_odata_response({"value": []}, _target()) == []
    assert replay.normalize_traceability_response({"items": []}, _target()) == []


def test_normalize_gee_maps_canonical_fields() -> None:
    raw = {
        "features": [
            {
                "id": "S2_SCENE",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
                "properties": {
                    "PRODUCT_ID": "S2A_MSIL2A_20220103",
                    "datetime": "2022-01-03T13:00:00Z",
                    "MGRS_TILE": "25MGR",
                },
            }
        ]
    }
    results = replay.normalize_gee_response(raw, _target())
    assert len(results) == 1
    assert results[0].product_id == "S2A_MSIL2A_20220103"
    assert results[0].provider == "GEE"
    assert results[0].source_response_hash


def test_normalize_odata_never_publishes_s3path() -> None:
    raw = {
        "value": [
            {
                "Id": "odata-1",
                "Name": "S2A_MSIL2A.SAFE",
                "S3Path": "/eodata/secret/path",
                "GeoFootprint": {"type": "Polygon", "coordinates": [[[0, 0]]]},
                "ContentDate": {"Start": "2022-01-03T13:00:00Z"},
                "Attributes": [{"Name": "tileId", "Value": "25MGR"}],
            }
        ]
    }
    results = replay.normalize_odata_response(raw, _target())
    assert results[0].odata_s3path == ""
    assert results[0].odata_id == "odata-1"


def test_load_replay_fixture_missing_returns_empty(tmp_path: Path) -> None:
    assert replay.load_replay_fixture("GEE", fixture_dir=tmp_path) == {}
    assert replay.load_replay_fixture("UNKNOWN", fixture_dir=tmp_path) == {}


def test_write_fixtures_creates_empty_public_payloads(tmp_path: Path) -> None:
    replay.write_fixtures(fixture_dir=tmp_path)
    gee = json.loads((tmp_path / "gee_empty_result.json").read_text(encoding="utf-8"))
    odata = json.loads((tmp_path / "odata_empty_result.json").read_text(encoding="utf-8"))
    assert gee["features"] == []
    assert odata["value"] == []
