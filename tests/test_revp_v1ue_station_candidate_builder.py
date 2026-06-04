"""Tests for v1ue — Station Candidate Builder."""

import csv
import os
import subprocess
import sys

import pytest

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ue_station_candidate_builder.py")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
POLICY = os.path.join("configs", "protocolo_c", "v1ue_station_search_policy.yaml")

STATION_COLUMNS = [
    "station_candidate_id", "source_id", "event_id", "city", "uf",
    "station_name", "station_code", "station_type", "latitude", "longitude",
    "coordinate_source", "coordinate_status", "distance_to_city_km",
    "distance_method", "is_official", "can_anchor_temporal_evidence",
    "can_anchor_spatial_evidence", "limitations", "acquisition_status", "notes",
]


@pytest.fixture
def built_stations(tmp_path):
    out = str(tmp_path / "stations.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--events", EVENTS, "--policy-config", POLICY, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out


class TestStationBuilder:
    def test_dry_run(self, tmp_path):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--events", EVENTS, "--policy-config", POLICY, "--dry-run"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout

    def test_required_columns(self, built_stations):
        with open(built_stations, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in STATION_COLUMNS:
            assert col in cols

    def test_station_not_flood_geometry(self, built_stations):
        with open(built_stations, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_anchor_spatial_evidence"] == "false"

    def test_station_can_anchor_temporal(self, built_stations):
        with open(built_stations, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["can_anchor_temporal_evidence"] == "true"

    def test_missing_coordinate_marked(self, built_stations):
        with open(built_stations, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if not r["latitude"]:
                assert r["coordinate_status"] == "MISSING"

    def test_no_invented_coordinates(self, built_stations):
        with open(built_stations, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["coordinate_status"] == "MISSING":
                assert r["latitude"] == ""
                assert r["longitude"] == ""

    def test_no_absolute_paths(self, built_stations):
        with open(built_stations, "r", encoding="utf-8") as f:
            content = f.read()
        assert "C:\\Users" not in content
        assert "/home/" not in content
