"""Tests for v1ue — Official Dataset Resolver."""

import csv
import os
import subprocess
import sys

import pytest

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ue_official_dataset_resolver.py")
TARGETS = os.path.join("configs", "protocolo_c", "v1ue_official_dataset_targets.yaml")
SOURCES = os.path.join("configs", "protocolo_c", "ground_reference_evidence_sources.yaml")
DOMAINS = os.path.join("configs", "protocolo_c", "v1ud_allowed_domains.yaml")

RESOLUTION_COLUMNS = [
    "dataset_resolution_id", "event_id", "source_id", "target_year",
    "target_city", "target_hazard", "query_terms", "candidate_url",
    "http_status", "content_type", "content_length", "dataset_type",
    "is_event_specific", "is_year_specific", "is_city_specific",
    "is_downloadable", "license_status", "resolution_status", "blocking_reason",
]


@pytest.fixture
def dry_resolution(tmp_path):
    out = str(tmp_path / "resolution.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--targets-config", TARGETS,
         "--sources-config", SOURCES, "--domains-config", DOMAINS,
         "--out", out, "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out


class TestDatasetResolver:
    def test_dry_run_runs(self, dry_resolution):
        assert os.path.exists(dry_resolution)
        with open(dry_resolution, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

    def test_required_columns(self, dry_resolution):
        with open(dry_resolution, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in RESOLUTION_COLUMNS:
            assert col in cols

    def test_dry_run_all_dry_status(self, dry_resolution):
        with open(dry_resolution, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["http_status"] == "DRY_RUN"

    def test_year_specific_classification(self, dry_resolution):
        with open(dry_resolution, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # INMET year zips have year in URL -> is_year_specific true
        inmet = [r for r in rows if r["source_id"] == "INMET_BDMEP"]
        assert any(r["is_year_specific"] == "true" for r in inmet)

    def test_generic_homepage_not_event_specific(self, dry_resolution):
        with open(dry_resolution, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Cemaden mapainterativo homepage has no year/city in URL
        cemaden = [r for r in rows if r["source_id"] == "CEMADEN_PLUVIOMETROS"]
        for r in cemaden:
            assert r["is_event_specific"] == "false"

    def test_susceptibility_source_needs_review(self, dry_resolution):
        with open(dry_resolution, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # SGB/CPRM has UNKNOWN license -> blocking mentions LICENSE_NEEDS_REVIEW
        sgb = [r for r in rows if r["source_id"] == "SGB_CPRM_CARTOGRAFIA"]
        for r in sgb:
            assert "LICENSE_NEEDS_REVIEW" in r["blocking_reason"] or r["license_status"] != "PUBLIC_OPEN"
