"""Tests for v1ui — Public Portal Crawler."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_public_portal_crawler.py")
FIXTURES = os.path.join("tests", "fixtures", "v1ui")


def _make_discovery(tmp_path, entries):
    path = os.path.join(tmp_path, "discovery.csv")
    cols = ["discovery_id", "event_id", "source_id", "candidate_url",
            "candidate_class", "region", "source_name"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(entries)
    return path


def _run(tmp_path, discovery_entries=None):
    if discovery_entries is None:
        discovery_entries = [{"discovery_id": "D1", "event_id": "PET_2022_02_15",
                              "source_id": "S1", "candidate_url": "https://example.gov.br/",
                              "candidate_class": "CITY_PORTAL_PUBLIC_CANDIDATE",
                              "region": "PET", "source_name": "Test"}]
    disc_path = _make_discovery(tmp_path, discovery_entries)
    out = os.path.join(tmp_path, "crawl.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--discovery", disc_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPublicPortalCrawler:
    def test_dry_run_produces_output(self, tmp_path):
        rows = _run(str(tmp_path))
        assert len(rows) >= 1

    def test_dry_run_status(self, tmp_path):
        rows = _run(str(tmp_path))
        assert rows[0]["artifact_candidate_status"] == "DRY_RUN"

    def test_crawl_ids_unique(self, tmp_path):
        rows = _run(str(tmp_path), [
            {"discovery_id": "D1", "event_id": "E1", "source_id": "S1",
             "candidate_url": "https://a.gov.br/", "candidate_class": "C",
             "region": "", "source_name": ""},
            {"discovery_id": "D2", "event_id": "E2", "source_id": "S2",
             "candidate_url": "https://b.gov.br/", "candidate_class": "C",
             "region": "", "source_name": ""},
        ])
        ids = [r["crawl_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_link_extractor_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ui_public_portal_crawler import LinkExtractor
        html = '<a href="/data.zip">Download ZIP</a><a href="/report.pdf">Report</a>'
        ext = LinkExtractor()
        ext.feed(html)
        assert len(ext.links) == 2
        assert ext.links[0] == ("/data.zip", "Download ZIP")

    def test_score_link_artifact_extension(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ui_public_portal_crawler import score_link
        score, _, _, _ = score_link("https://example.com/data.geojson", "data", {})
        assert score >= 30

    def test_score_link_arcgis_service(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ui_public_portal_crawler import score_link
        score, _, _, _ = score_link("https://example.com/arcgis/rest/MapServer", "", {})
        assert score >= 25
