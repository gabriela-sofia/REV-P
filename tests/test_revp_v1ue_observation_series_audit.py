"""Tests for v1ue — Observation Series Audit."""

import csv
import os
import subprocess
import sys
import tempfile

import pytest

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ue_observation_series_audit.py")
EXTRACTION = os.path.join("datasets", "protocolo_c", "v1ud_evidence_extraction_registry.csv")
STATIONS = os.path.join("datasets", "protocolo_c", "v1ue_station_candidate_registry.csv")

OBSERVATION_COLUMNS = [
    "observation_id", "event_id", "source_id", "station_candidate_id",
    "asset_path_hash", "asset_type", "observed_variable", "observed_start",
    "observed_end", "temporal_overlap_status", "precipitation_total_mm",
    "water_level_signal", "discharge_signal", "hazard_terms_found",
    "locality_terms_found", "geometry_metadata_available", "event_specificity",
    "evidence_role", "evidence_strength", "gate_support", "limitations", "notes",
]


@pytest.fixture
def audit_output(tmp_path):
    out = str(tmp_path / "observations.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--extraction-registry", EXTRACTION,
         "--stations", STATIONS, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out


class TestObservationAudit:
    def test_runs(self, audit_output):
        assert os.path.exists(audit_output)

    def test_required_columns(self, audit_output):
        with open(audit_output, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in OBSERVATION_COLUMNS:
            assert col in cols

    def test_html_portal_does_not_close_event_gate(self, audit_output):
        with open(audit_output, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["asset_type"] == "HTML":
                assert r["gate_support"] == "none_generic_portal"
                assert r["geometry_metadata_available"] == "false"

    def test_no_overlay_in_limitations(self, audit_output):
        with open(audit_output, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert "overlay" in r["limitations"].lower() or r["asset_type"] == "GEODATA"


class TestProbeHelpers:
    def test_audit_html_detects_terms(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ue_observation_series_audit import audit_html

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
            f.write("<html><body>Houve inundacao no bairro central</body></html>")
            path = f.name
        try:
            result = audit_html(path)
            assert result["asset_type"] == "HTML"
            assert "inundacao" in result["hazard_terms"]
            assert result["evidence_role"] == "contextual_only"
        finally:
            os.unlink(path)

    def test_audit_pdf_missing_backend_or_error(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ue_observation_series_audit import audit_pdf

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"not a real pdf")
            path = f.name
        try:
            result = audit_pdf(path)
            assert result["probe"] in ("PDF_BACKEND_MISSING", "PDF_PARSE_ERROR")
        finally:
            os.unlink(path)
