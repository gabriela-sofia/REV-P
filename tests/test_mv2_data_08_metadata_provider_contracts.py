from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_08_metadata_provider_contracts as c


def _target(**kwargs) -> c.MetadataQueryTarget:
    base = dict(
        patch_id="P1",
        asset_id="A1",
        sensor_family="SENTINEL_2",
        temporal_window_start="2022-01-01",
        temporal_window_end="2022-01-10",
        aoi_wgs84=[-35.0, -8.1, -34.9, -8.0],
    )
    base.update(kwargs)
    return c.MetadataQueryTarget(**base)


def test_valid_target_is_query_ready() -> None:
    assert _target().is_query_ready() is True


def test_invalid_temporal_window_blocks() -> None:
    assert _target(temporal_window_start="2022-01-10", temporal_window_end="2022-01-01").has_valid_temporal_window() is False
    assert _target(temporal_window_start="", temporal_window_end="").has_valid_temporal_window() is False


def test_non_sentinel_2_blocks() -> None:
    assert _target(sensor_family="LANDSAT_8").is_sentinel_2() is False
    assert _target(sensor_family="").is_query_ready() is False


def test_aoi_validation_accepts_bbox_and_geometry() -> None:
    assert _target(aoi_wgs84=[-35.0, -8.1, -34.9, -8.0]).has_valid_aoi() is True
    assert _target(aoi_wgs84={"type": "Polygon", "coordinates": [[[0, 0]]]}).has_valid_aoi() is True
    assert _target(aoi_wgs84=None).has_valid_aoi() is False
    assert _target(aoi_wgs84=[1, 2, 3]).has_valid_aoi() is False


def test_result_default_is_no_call() -> None:
    result = c.MetadataProviderResult(patch_id="P1", asset_id="A1", provider="GEE")
    assert result.status == "NO_CALL"
    assert result.is_call_executed() is False
    assert result.has_product_match() is False


def test_result_rejects_invalid_status() -> None:
    with pytest.raises(ValueError):
        c.MetadataProviderResult(patch_id="P1", asset_id="A1", provider="GEE", status="BOGUS")


def test_result_rejects_invalid_query_mode() -> None:
    with pytest.raises(ValueError):
        c.MetadataProviderResult(patch_id="P1", asset_id="A1", provider="GEE", query_mode="RASTER")


def test_result_to_row_has_all_required_fields() -> None:
    result = c.MetadataProviderResult(patch_id="P1", asset_id="A1", provider="GEE")
    row = result.to_row()
    assert set(row.keys()) == set(c.RESULT_FIELDS)


def test_match_status_marks_call_executed() -> None:
    result = c.MetadataProviderResult(
        patch_id="P1", asset_id="A1", provider="GEE", status="MATCH_STRONG", product_id="S2_X"
    )
    assert result.is_call_executed() is True
    assert result.has_product_match() is True


def test_no_call_helper() -> None:
    result = c.no_call_result("P1", "A1", "CDSE_STAC", "live_disabled")
    assert result.status == "NO_CALL"
    assert result.blocked_reason == "live_disabled"


def test_no_contract_field_references_raster_or_crop() -> None:
    forbidden = {"raster", "crop", ".tif", "geotiff", "local_path", "signed_url", "download"}
    for field in c.RESULT_FIELDS:
        assert all(token not in field for token in forbidden)


def test_consensus_record_serialises_lists_as_strings() -> None:
    record = c.MetadataConsensusRecord(
        patch_id="P1",
        asset_id="A1",
        consensus_status="STRONG",
        providers_considered=["GEE", "CDSE_STAC"],
        providers_agreeing=["GEE", "CDSE_STAC"],
    )
    row = record.to_row()
    assert row["providers_considered"] == "GEE;CDSE_STAC"
    assert row["consensus_status"] == "STRONG"
