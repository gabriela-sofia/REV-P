"""Tests for v1uh — Formal Response Intake."""

import csv
import os
import shutil
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uh_formal_response_intake.py")
FIXTURES = os.path.join("tests", "fixtures", "v1uh")

RESPONSE_COLUMNS = [
    "response_id", "institution", "event_id", "received_date",
    "source_channel", "original_filename", "local_raw_path_hash",
    "sha256", "file_size_bytes", "mime_type", "extension",
    "intake_status", "quarantine_reason", "sensitive_review_required",
    "license_status", "redistribution_allowed", "notes",
]


def _run(tmp_path, inbox_files=None):
    inbox = os.path.join(tmp_path, "inbox")
    staging = os.path.join(tmp_path, "staging")
    quarantine = os.path.join(tmp_path, "quarantine")
    out_dir = os.path.join(tmp_path, "out")
    os.makedirs(inbox, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if inbox_files:
        for src in inbox_files:
            shutil.copy2(src, inbox)

    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--inbox", inbox, "--staging", staging,
         "--quarantine", quarantine, "--out-dir", out_dir],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return os.path.join(out_dir, "v1uh_formal_response_registry.csv")


class TestFormalResponseIntake:
    def test_empty_inbox_no_crash(self, tmp_path):
        out = _run(str(tmp_path))
        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_empty_inbox_valid_columns(self, tmp_path):
        out = _run(str(tmp_path))
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in RESPONSE_COLUMNS:
            assert col in cols

    def test_csv_accepted(self, tmp_path):
        out = _run(str(tmp_path), [os.path.join(FIXTURES, "synthetic_occurrences.csv")])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["intake_status"] == "ACCEPTED"
        assert rows[0]["extension"] == ".csv"

    def test_geojson_accepted(self, tmp_path):
        out = _run(str(tmp_path), [os.path.join(FIXTURES, "synthetic_geometry.geojson")])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["intake_status"] == "ACCEPTED"

    def test_zip_accepted(self, tmp_path):
        out = _run(str(tmp_path), [os.path.join(FIXTURES, "synthetic_package.zip")])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["intake_status"] == "ACCEPTED"

    def test_sha256_computed(self, tmp_path):
        out = _run(str(tmp_path), [os.path.join(FIXTURES, "synthetic_occurrences.csv")])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows[0]["sha256"]) == 64

    def test_suspicious_extension_quarantined(self, tmp_path):
        bad_file = os.path.join(str(tmp_path), "evil.exe")
        with open(bad_file, "w") as f:
            f.write("not really")
        out = _run(str(tmp_path), [bad_file])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["intake_status"] == "QUARANTINED"
        assert rows[0]["quarantine_reason"] == "suspicious_extension"

    def test_sensitive_filename_quarantined(self, tmp_path):
        sens_file = os.path.join(str(tmp_path), "cadastro_vitimas_cpf.csv")
        with open(sens_file, "w") as f:
            f.write("cpf,nome\n12345678901,Test")
        out = _run(str(tmp_path), [sens_file])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["intake_status"] == "QUARANTINED"

    def test_staging_receives_accepted(self, tmp_path):
        _run(str(tmp_path), [os.path.join(FIXTURES, "synthetic_occurrences.csv")])
        staging = os.path.join(str(tmp_path), "staging")
        assert os.path.exists(os.path.join(staging, "synthetic_occurrences.csv"))

    def test_original_not_deleted(self, tmp_path):
        inbox = os.path.join(str(tmp_path), "inbox")
        os.makedirs(inbox, exist_ok=True)
        src = os.path.join(FIXTURES, "synthetic_occurrences.csv")
        shutil.copy2(src, inbox)
        _run(str(tmp_path), None)
        assert os.path.exists(os.path.join(inbox, "synthetic_occurrences.csv"))

    def test_classify_file_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uh_formal_response_intake import classify_file
        csv_path = os.path.join(FIXTURES, "synthetic_occurrences.csv")
        status, reason = classify_file(csv_path, "synthetic_occurrences.csv")
        assert status == "ACCEPTED"
        assert reason == ""
