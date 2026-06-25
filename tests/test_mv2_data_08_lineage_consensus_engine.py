from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import mv2_data_08_lineage_consensus_engine as engine
from mv2_data_08_metadata_provider_contracts import MetadataProviderResult

GEOM = {"type": "Polygon", "coordinates": [[[0, 0]]]}


def _result(provider: str, **kwargs) -> MetadataProviderResult:
    base = dict(patch_id="P1", asset_id="A1", provider=provider, status="MATCH_STRONG")
    base.update(kwargs)
    return MetadataProviderResult(**base)


def test_no_call_when_all_results_blocked() -> None:
    results = [
        MetadataProviderResult(patch_id="P1", asset_id="A1", provider="GEE", status="NO_CALL"),
        MetadataProviderResult(patch_id="P1", asset_id="A1", provider="CDSE_STAC", status="BLOCKED_BY_FLAGS"),
    ]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "NO_CALL"


def test_no_match_when_call_made_but_no_product() -> None:
    results = [MetadataProviderResult(patch_id="P1", asset_id="A1", provider="GEE", status="NO_MATCH")]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "NO_MATCH"


def test_strong_when_two_official_sources_agree() -> None:
    results = [
        _result("GEE", product_id="S2A_X", datetime_utc="2022-01-03T13:00:00Z", geometry=GEOM),
        _result("CDSE_STAC", product_id="S2A_X", datetime_utc="2022-01-03T13:00:00Z", geometry=GEOM),
    ]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "STRONG"
    assert record.product_id == "S2A_X"
    assert "GEE" in record.providers_agreeing


def test_strong_when_single_official_with_product_datetime_geometry() -> None:
    results = [_result("CDSE_ODATA", product_id="S2A_X", datetime_utc="2022-01-03T13:00:00Z", odata_geofootprint=GEOM)]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "STRONG"


def test_medium_review_when_no_geometry_but_tile_and_collection() -> None:
    # datetime + collection + mgrs but geometry missing -> not strong, weak_blocked
    results = [_result("GEE", product_id="S2A_X", datetime_utc="2022-01-03T13:00:00Z", mgrs_tile="25MGR", collection="S2_L2A")]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "WEAK_BLOCKED"


def test_medium_review_with_geometry_collection_tile_but_single_weak_source() -> None:
    results = [
        _result("GEE", status="MATCH_MEDIUM_REVIEW", product_id="S2A_X", datetime_utc="2022-01-03T13:00:00Z",
                mgrs_tile="25MGR", collection="S2_L2A", geometry=GEOM),
    ]
    # single official source WITH geometry+datetime+product -> STRONG per rule 2
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "STRONG"


def test_weak_blocked_when_no_geometry() -> None:
    results = [_result("GEE", status="MATCH_WEAK", product_id="S2A_X", mgrs_tile="25MGR")]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "WEAK_BLOCKED"


def test_conflict_on_divergent_product_id() -> None:
    results = [
        _result("GEE", product_id="S2A_X", datetime_utc="2022-01-03T13:00:00Z", geometry=GEOM),
        _result("CDSE_STAC", product_id="S2A_Y", datetime_utc="2022-01-03T13:00:00Z", geometry=GEOM),
    ]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "CONFLICT"
    assert record.conflict_reason == "divergent_product_id"


def test_conflict_on_incompatible_datetime() -> None:
    results = [
        _result("GEE", product_id="S2A_X", datetime_utc="2022-01-03T13:00:00Z", geometry=GEOM),
        _result("CDSE_STAC", product_id="S2A_X", datetime_utc="2022-01-09T13:00:00Z", geometry=GEOM),
    ]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "CONFLICT"
    assert record.conflict_reason == "incompatible_datetime"


def test_conflict_when_provider_reports_conflict() -> None:
    results = [_result("GEE", status="CONFLICT")]
    record = engine.compute_consensus("P1", "A1", results)
    assert record.consensus_status == "CONFLICT"


def test_summary_counts_statuses() -> None:
    records = [
        engine.compute_consensus("P1", "A1", [_result("GEE", product_id="X", datetime_utc="2022-01-03", geometry=GEOM),
                                              _result("CDSE_STAC", product_id="X", datetime_utc="2022-01-03", geometry=GEOM)]),
    ]
    summary = engine.summarize(records)
    assert summary["counts"]["STRONG"] == 1
    assert summary["confirmed_lineage"] == 1
    assert summary["live_calls"] == 0


def test_write_outputs(tmp_path: Path) -> None:
    records = [engine.compute_consensus("P1", "A1", [_result("GEE", status="NO_CALL")])]
    paths = engine.write_outputs(records, out_dir=tmp_path)
    assert paths["csv"].exists()
    assert paths["summary"].exists()
    assert paths["conflict"].exists()
