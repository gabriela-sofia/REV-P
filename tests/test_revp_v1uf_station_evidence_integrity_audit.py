"""Tests for v1uf — Station Evidence Integrity Audit."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uf_station_evidence_integrity_audit.py")
MANIFEST = os.path.join("datasets", "protocolo_c", "v1uf_large_download_manifest.csv")
ASSETS = os.path.join("datasets", "protocolo_c", "v1uf_station_series_asset_registry.csv")
CATALOG = os.path.join("datasets", "protocolo_c", "v1uf_official_station_catalog_registry.csv")
METRICS = os.path.join("datasets", "protocolo_c", "v1uf_hydromet_window_metrics_registry.csv")

AUDIT_COLUMNS = [
    "audit_id", "event_id", "station_candidate_id", "asset_id",
    "check_name", "status", "severity", "reason", "required_action",
]


def _run(tmp_path):
    out = str(tmp_path / "out")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--manifest", MANIFEST, "--assets", ASSETS,
         "--catalog", CATALOG, "--metrics", METRICS, "--out-dir", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return os.path.join(out, "v1uf_station_evidence_integrity_registry.csv")


class TestIntegrityAudit:
    def test_runs_and_columns(self, tmp_path):
        reg = _run(tmp_path)
        assert os.path.exists(reg)
        with open(reg, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in AUDIT_COLUMNS:
            assert col in cols

    def test_all_status_valid(self, tmp_path):
        reg = _run(tmp_path)
        valid = {"PASS", "FAIL", "NEEDS_REVIEW", "NOT_APPLICABLE"}
        with open(reg, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["status"] in valid

    def test_spatial_truth_misuse_check_present(self, tmp_path):
        """The CRITICAL spatial-truth-misuse guardrail check must exist."""
        reg = _run(tmp_path)
        with open(reg, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        spatial = [r for r in rows if r["check_name"] == "spatial_truth_misuse_risk"]
        assert len(spatial) > 0
        for r in spatial:
            assert r["severity"] == "CRITICAL"
            assert r["required_action"] == "DO_NOT_PROMOTE_GROUND_REFERENCE"

    def test_event_vs_year_specificity_flagged(self, tmp_path):
        reg = _run(tmp_path)
        with open(reg, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        spec = [r for r in rows if r["check_name"] == "event_vs_year_specificity"]
        # If any asset extracted, this check should be NEEDS_REVIEW (year not event specific)
        for r in spec:
            assert r["status"] == "NEEDS_REVIEW"
