"""Tests for v1ud — Raw Asset Integrity Audit."""

import csv
import hashlib
import os
import subprocess
import sys
import tempfile

import pytest

AUDIT_SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ud_raw_asset_integrity_audit.py")
ACQ_SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ud_real_source_acquisition.py")
RESOLVER = os.path.join("scripts", "protocolo_c", "revp_v1ud_source_url_resolver.py")
MANIFEST_SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ud_download_manifest_builder.py")

SOURCES_CONFIG = os.path.join("configs", "protocolo_c", "ground_reference_evidence_sources.yaml")
TARGETS_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_real_acquisition_targets.yaml")
DOMAINS_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_allowed_domains.yaml")
POLICY_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_download_policy.yaml")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
V1UC_EVIDENCE = os.path.join("datasets", "protocolo_c", "evidence_source_registry.csv")


@pytest.fixture
def dry_run_pipeline(tmp_path):
    out_dir = str(tmp_path / "datasets")
    local_dir = str(tmp_path / "local")

    res_out = os.path.join(out_dir, "resolution.csv")
    subprocess.run(
        [
            sys.executable, RESOLVER,
            "--sources-config", SOURCES_CONFIG,
            "--targets-config", TARGETS_CONFIG,
            "--domains-config", DOMAINS_CONFIG,
            "--out", res_out,
            "--dry-run",
        ],
        capture_output=True, text=True, timeout=60,
    )

    man_out = os.path.join(out_dir, "manifest.csv")
    subprocess.run(
        [
            sys.executable, MANIFEST_SCRIPT,
            "--resolution-registry", res_out,
            "--policy-config", POLICY_CONFIG,
            "--domains-config", DOMAINS_CONFIG,
            "--local-only-dir", local_dir,
            "--out", man_out,
        ],
        capture_output=True, text=True, timeout=60,
    )

    subprocess.run(
        [
            sys.executable, ACQ_SCRIPT,
            "--manifest", man_out,
            "--out-dir", out_dir,
            "--local-only-dir", local_dir,
            "--dry-run",
        ],
        capture_output=True, text=True, timeout=60,
    )

    return {
        "extraction_registry": os.path.join(out_dir, "v1ud_evidence_extraction_registry.csv"),
        "out_dir": out_dir,
    }


class TestIntegrityAudit:
    def test_audit_runs(self, dry_run_pipeline, tmp_path):
        int_out = str(tmp_path / "integrity.csv")
        delta_out = str(tmp_path / "delta.csv")
        result = subprocess.run(
            [
                sys.executable, AUDIT_SCRIPT,
                "--extraction-registry", dry_run_pipeline["extraction_registry"],
                "--v1uc-evidence", V1UC_EVIDENCE,
                "--out-integrity", int_out,
                "--out-gate-delta", delta_out,
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        assert os.path.exists(int_out)
        assert os.path.exists(delta_out)

    def test_integrity_has_columns(self, dry_run_pipeline, tmp_path):
        int_out = str(tmp_path / "integrity.csv")
        delta_out = str(tmp_path / "delta.csv")
        subprocess.run(
            [
                sys.executable, AUDIT_SCRIPT,
                "--extraction-registry", dry_run_pipeline["extraction_registry"],
                "--v1uc-evidence", V1UC_EVIDENCE,
                "--out-integrity", int_out,
                "--out-gate-delta", delta_out,
            ],
            capture_output=True, text=True, timeout=60,
        )
        required = ["integrity_id", "extraction_id", "source_id", "file_exists", "hash_match"]
        with open(int_out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in required:
            assert col in cols

    def test_gate_delta_no_ground_reference(self, dry_run_pipeline, tmp_path):
        int_out = str(tmp_path / "integrity.csv")
        delta_out = str(tmp_path / "delta.csv")
        subprocess.run(
            [
                sys.executable, AUDIT_SCRIPT,
                "--extraction-registry", dry_run_pipeline["extraction_registry"],
                "--v1uc-evidence", V1UC_EVIDENCE,
                "--out-integrity", int_out,
                "--out-gate-delta", delta_out,
            ],
            capture_output=True, text=True, timeout=60,
        )
        with open(delta_out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row.get("can_create_ground_reference") == "false"


class TestSHA256Integrity:
    def test_hash_computation(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ud_raw_asset_integrity_audit import sha256_file

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"integrity test content v1ud")
            path = f.name
        try:
            result = sha256_file(path)
            expected = hashlib.sha256(b"integrity test content v1ud").hexdigest()
            assert result == expected
        finally:
            os.unlink(path)


class TestProbes:
    def test_probe_pdf_missing_backend(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ud_raw_asset_integrity_audit import probe_pdf

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"not a real pdf")
            path = f.name
        try:
            result = probe_pdf(path)
            assert result["status"] in ("PDF_BACKEND_MISSING", "PDF_PARSE_ERROR")
        finally:
            os.unlink(path)

    def test_probe_zip_invalid(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ud_raw_asset_integrity_audit import probe_zip

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as f:
            f.write(b"not a real zip")
            path = f.name
        try:
            result = probe_zip(path)
            assert result["status"] == "ZIP_ERROR"
        finally:
            os.unlink(path)
