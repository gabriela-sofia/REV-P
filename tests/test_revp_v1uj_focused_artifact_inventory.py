"""Tests for v1uj — Focused Artifact Inventory."""

import csv
import os
import shutil
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_focused_artifact_inventory.py")
ZIP_FIXTURE = os.path.join("tests", "fixtures", "v1uj", "synthetic_focused_package.zip")

GEOJSON = ('{"type":"FeatureCollection",'
           '"crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:EPSG::4326"}},'
           '"features":[{"type":"Feature","properties":{},'
           '"geometry":{"type":"Point","coordinates":[-34.88,-8.05]}}]}')


def _run(tmp_path, raw_dir):
    out = os.path.join(tmp_path, "inventory.csv")
    cmd = [sys.executable, SCRIPT, "--raw-dir", raw_dir, "--out", out]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestFocusedArtifactInventory:
    def test_empty_dir(self, tmp_path):
        raw = os.path.join(str(tmp_path), "raw")
        os.makedirs(raw, exist_ok=True)
        assert _run(str(tmp_path), raw) == []

    def test_zip_with_vector_detected(self, tmp_path):
        ev = os.path.join(str(tmp_path), "raw", "rigeo", "PET_2022_02_15")
        os.makedirs(ev, exist_ok=True)
        shutil.copy(ZIP_FIXTURE, ev)
        rows = _run(str(tmp_path), os.path.join(str(tmp_path), "raw"))
        geo = [r for r in rows if r["extension"] == ".geojson"]
        assert geo
        assert geo[0]["has_geometry"] == "true"
        assert geo[0]["geometry_type"] == "Point"
        assert "4326" in geo[0]["crs"]
        assert any(r["source_tag"] == "rigeo" for r in rows)
        assert any(r["event_id"] == "PET_2022_02_15" for r in rows)

    def test_standalone_observed_geometry(self, tmp_path):
        ev = os.path.join(str(tmp_path), "raw", "ckan", "REC_2022_05_24_30")
        os.makedirs(ev, exist_ok=True)
        with open(os.path.join(ev, "ocorrencias_inundacao_recife_2022.geojson"),
                  "w", encoding="utf-8") as f:
            f.write(GEOJSON)
        rows = _run(str(tmp_path), os.path.join(str(tmp_path), "raw"))
        assert any(r["classification"] == "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW" for r in rows)

    def test_classify_asset_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uj_focused_artifact_inventory import (
            classify_asset, parse_geojson_meta)
        assert classify_asset("geospatial_vector", True, False, True) == "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW"
        assert classify_asset("geospatial_vector", False, True, True) == "CONTEXT_ONLY"
        assert classify_asset("document", False, False, False) == "document_only"
        gtype, crs, n = parse_geojson_meta(GEOJSON.encode("utf-8"))
        assert gtype == "Point" and "4326" in crs and n == "1"
