"""Tests for v1ui — ArcGIS / GeoServer Resolver."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_arcgis_geoserver_resolver.py")


def _make_inputs(tmp_path, discoveries, crawl=None):
    disc_path = os.path.join(tmp_path, "discovery.csv")
    disc_cols = ["discovery_id", "event_id", "source_id", "candidate_url", "candidate_class"]
    with open(disc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=disc_cols)
        writer.writeheader()
        writer.writerows(discoveries)

    crawl_path = os.path.join(tmp_path, "crawl.csv")
    crawl_cols = ["crawl_id", "event_id", "source_id", "discovered_url", "detected_service_type"]
    with open(crawl_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=crawl_cols)
        writer.writeheader()
        if crawl:
            writer.writerows(crawl)

    return disc_path, crawl_path


def _run(tmp_path, discoveries, crawl=None):
    disc_path, crawl_path = _make_inputs(tmp_path, discoveries, crawl)
    out = os.path.join(tmp_path, "layers.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--discovery", disc_path,
         "--crawl-manifest", crawl_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestArcgisGeoserverResolver:
    def test_empty_input(self, tmp_path):
        rows = _run(str(tmp_path), [])
        assert len(rows) == 0

    def test_arcgis_candidate_dry_run(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "discovery_id": "D1", "event_id": "PET_2022_02_15",
            "source_id": "SGB", "candidate_url": "https://geosgb.sgb.gov.br/MapServer",
            "candidate_class": "ARCGIS_REST_CANDIDATE",
        }])
        assert len(rows) >= 1
        assert rows[0]["candidate_status"] == "DRY_RUN"
        assert rows[0]["service_type"] == "arcgis_rest"

    def test_service_ids_unique(self, tmp_path):
        rows = _run(str(tmp_path), [
            {"discovery_id": "D1", "event_id": "E1", "source_id": "S1",
             "candidate_url": "https://a.gov.br/MapServer",
             "candidate_class": "ARCGIS_REST_CANDIDATE"},
            {"discovery_id": "D2", "event_id": "E2", "source_id": "S2",
             "candidate_url": "https://b.gov.br/MapServer",
             "candidate_class": "ARCGIS_REST_CANDIDATE"},
        ])
        ids = [r["service_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_crawl_detected_service_included(self, tmp_path):
        rows = _run(str(tmp_path), [], [{
            "crawl_id": "C1", "event_id": "E1", "source_id": "S1",
            "discovered_url": "https://geosgb.sgb.gov.br/arcgis/rest/FeatureServer",
            "detected_service_type": "arcgis_featureserver",
        }])
        assert len(rows) >= 1
