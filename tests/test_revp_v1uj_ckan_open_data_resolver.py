"""Tests for v1uj — CKAN Open Data Resolver."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_ckan_open_data_resolver.py")
FIXTURE = os.path.join("tests", "fixtures", "v1uj", "synthetic_ckan_package_search.json")


def _run(tmp_path, extra=None):
    out = os.path.join(tmp_path, "ckan.csv")
    cmd = [sys.executable, SCRIPT, "--out", out] + (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestCkanOpenDataResolver:
    def test_dry_run_no_web(self, tmp_path):
        rows = _run(str(tmp_path))
        assert len(rows) >= 1
        assert any(r["blocking_reason"] == "DRY_RUN" for r in rows)

    def test_fixture_geospatial_candidate(self, tmp_path):
        rows = _run(str(tmp_path), ["--search-fixture", FIXTURE])
        geo = [r for r in rows if r["is_geospatial_candidate"] == "true"]
        assert geo
        assert any(r["resource_format"].upper() == "GEOJSON" for r in geo)

    def test_drainage_is_contextual(self, tmp_path):
        rows = _run(str(tmp_path), ["--search-fixture", FIXTURE])
        ctx = [r for r in rows if r["is_contextual_only"] == "true"]
        assert ctx, "drenagem package should be contextual_only"
        assert all(r["blocking_reason"] == "generic_infrastructure_not_occurrence"
                   for r in ctx)

    def test_classify_resource_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uj_ckan_open_data_resolver import classify_resource
        cfg = {"geospatial_formats": ["GeoJSON", "SHP"],
               "tabular_formats": ["CSV"],
               "contextual_only_terms": ["drenagem"]}
        is_cand, is_ctx, prio = classify_resource("GeoJSON", "Ocorrencias alagamento", cfg)
        assert is_cand is True and is_ctx is False and prio == "1"
        is_cand2, is_ctx2, _ = classify_resource("SHP", "Rede de drenagem", cfg)
        assert is_cand2 is True and is_ctx2 is True
