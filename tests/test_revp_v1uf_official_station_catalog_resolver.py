"""Tests for v1uf — Official Station Catalog Resolver."""

import csv
import os
import subprocess
import sys

import pytest

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uf_official_station_catalog_resolver.py")
STATIONS = os.path.join("datasets", "protocolo_c", "v1ue_station_candidate_registry.csv")
CATALOG_CONFIG = os.path.join("configs", "protocolo_c", "v1uf_station_catalog_sources.yaml")
BINDING_CONFIG = os.path.join("configs", "protocolo_c", "v1uf_station_target_binding.yaml")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")

CATALOG_COLUMNS = [
    "station_candidate_id", "source_id", "official_catalog_id", "station_code",
    "station_name", "municipality", "uf", "latitude", "longitude",
    "altitude_m", "coordinate_status", "coordinate_source_url",
    "coordinate_source_sha256", "catalog_acquisition_status",
    "match_confidence", "match_method", "limitations",
    "can_anchor_temporal_evidence", "can_anchor_spatial_context",
    "can_create_ground_reference",
]


@pytest.fixture
def offline_resolution(tmp_path):
    """Run WITHOUT --allow-web: no catalog fetched -> coordinates stay MISSING."""
    out = str(tmp_path / "out")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--stations", STATIONS, "--catalog-config", CATALOG_CONFIG,
         "--binding-config", BINDING_CONFIG, "--events", EVENTS, "--out-dir", out,
         "--local-only-dir", str(tmp_path / "local"), "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return os.path.join(out, "v1uf_official_station_catalog_registry.csv")


class TestCatalogResolver:
    def test_required_columns(self, offline_resolution):
        with open(offline_resolution, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in CATALOG_COLUMNS:
            assert col in cols

    def test_offline_keeps_missing(self, offline_resolution):
        """Without web/catalog, every coordinate must remain MISSING (never invented)."""
        with open(offline_resolution, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["coordinate_status"] == "MISSING"
            assert r["latitude"] == ""
            assert r["longitude"] == ""

    def test_station_not_flood_geometry(self, offline_resolution):
        with open(offline_resolution, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_anchor_spatial_context"] == "false"
            assert r["can_create_ground_reference"] == "false"

    def test_match_function_requires_official_catalog(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uf_official_station_catalog_resolver import match_inmet_station
        catalog = [
            {"CD_ESTACAO": "A610", "VL_LATITUDE": "-22.46", "VL_LONGITUDE": "-43.29",
             "VL_ALTITUDE": "1777", "DC_NOME": "PICO DO COUTO"},
        ]
        m = match_inmet_station("A610", catalog)
        assert m["found"] is True
        assert m["latitude"] == "-22.46"
        # Unknown code -> not found, no coordinate
        m2 = match_inmet_station("ZZZZ", catalog)
        assert m2["found"] is False

    def test_no_geocoding_no_centroid(self, offline_resolution):
        """Coordinates must never come from city name/centroid."""
        with open(offline_resolution, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            # offline: source must not be a geocoder/centroid
            assert r["coordinate_source_url"] == ""
            assert r["match_method"] in ("NONE", "")
