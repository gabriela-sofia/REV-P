from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_08_metadata_clients as clients
from mv2_data_08_metadata_provider_contracts import MetadataQueryTarget

GOOD_CONFIG = {
    "allow_network": True,
    "allow_metadata_calls": True,
    "allow_raster_download": False,
    "allow_canary_download": False,
}


def _target(**kwargs) -> MetadataQueryTarget:
    base = dict(
        patch_id="P1",
        asset_id="A1",
        sensor_family="SENTINEL_2",
        temporal_window_start="2022-01-01",
        temporal_window_end="2022-01-10",
        aoi_wgs84=[-35.0, -8.1, -34.9, -8.0],
    )
    base.update(kwargs)
    return MetadataQueryTarget(**base)


def test_default_client_is_no_call() -> None:
    client = clients.GeeMetadataClient()
    result = client.query(_target())[0]
    assert result.status == "NO_CALL"
    assert client.call_count == 0


def test_gate_blocks_without_config_when_live() -> None:
    status, reason = clients.evaluate_call_gate(None, _target(), live=True)
    assert status == "BLOCKED_NO_CONFIG"


def test_gate_blocks_when_metadata_calls_disabled() -> None:
    status, _ = clients.evaluate_call_gate({"allow_network": False, "allow_metadata_calls": False}, _target(), live=True)
    assert status == "BLOCKED_BY_FLAGS"


def test_gate_blocks_when_raster_download_enabled() -> None:
    cfg = dict(GOOD_CONFIG, allow_raster_download=True)
    status, _ = clients.evaluate_call_gate(cfg, _target(), live=True)
    assert status == "BLOCKED_BY_FLAGS"


def test_gate_blocks_when_canary_download_enabled() -> None:
    cfg = dict(GOOD_CONFIG, allow_canary_download=True)
    status, _ = clients.evaluate_call_gate(cfg, _target(), live=True)
    assert status == "BLOCKED_BY_FLAGS"


def test_gate_blocks_without_temporal_window() -> None:
    status, _ = clients.evaluate_call_gate(GOOD_CONFIG, _target(temporal_window_start="", temporal_window_end=""), live=True)
    assert status == "BLOCKED_NO_TEMPORAL_WINDOW"


def test_gate_blocks_without_sentinel_2() -> None:
    status, _ = clients.evaluate_call_gate(GOOD_CONFIG, _target(sensor_family="LANDSAT_8"), live=True)
    assert status == "BLOCKED_NO_SENSOR_LINEAGE"


def test_gate_blocks_without_aoi() -> None:
    status, _ = clients.evaluate_call_gate(GOOD_CONFIG, _target(aoi_wgs84=None), live=True)
    assert status == "BLOCKED_NO_AOI"


def test_gate_ready_when_all_pass() -> None:
    status, reason = clients.evaluate_call_gate(GOOD_CONFIG, _target(), live=True)
    assert status == "QUERY_READY"
    assert reason == ""


def test_live_without_transport_is_query_ready_no_call() -> None:
    client = clients.GeeMetadataClient(live=True, config=GOOD_CONFIG)
    result = client.query(_target())[0]
    assert result.status == "QUERY_READY"
    assert client.call_count == 0  # no real network without injected transport


def test_live_with_mock_transport_normalizes_and_counts() -> None:
    raw = {
        "features": [
            {
                "id": "S2A_SCENE",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0]]]},
                "properties": {
                    "PRODUCT_ID": "S2A_MSIL2A_20220103",
                    "datetime": "2022-01-03T13:00:00Z",
                    "MGRS_TILE": "25MGR",
                    "CLOUDY_PIXEL_PERCENTAGE": 4.2,
                },
            }
        ]
    }
    client = clients.GeeMetadataClient(live=True, config=GOOD_CONFIG, transport=lambda t: raw)
    result = client.query(_target())[0]
    assert client.call_count == 1
    assert result.product_id == "S2A_MSIL2A_20220103"
    assert result.status == "MATCH_STRONG"
    assert result.query_mode == "METADATA_ONLY"


def test_live_transport_error_is_query_failed() -> None:
    def boom(_target):
        raise RuntimeError("network down")

    client = clients.CdseStacMetadataClient(live=True, config=GOOD_CONFIG, transport=boom)
    result = client.query(_target())[0]
    assert result.status == "QUERY_FAILED"
    assert client.call_count == 0


def test_replay_mode_uses_fixtures_no_network() -> None:
    client = clients.CdseODataMetadataClient(replay_mode=True, fixture_loader=lambda p: {"value": []})
    result = client.query(_target())[0]
    assert result.status == "NO_MATCH"
    assert result.query_mode == "REPLAY"
    assert client.call_count == 0


def test_build_clients_returns_all_providers() -> None:
    built = clients.build_clients(["GEE", "CDSE_STAC", "CDSE_ODATA", "TRACEABILITY"])
    assert set(built.keys()) == {"GEE", "CDSE_STAC", "CDSE_ODATA", "TRACEABILITY"}
