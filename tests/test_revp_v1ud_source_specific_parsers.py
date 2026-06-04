"""Tests for v1ud — Source-Specific Parsers."""

import csv
import os
import subprocess
import sys

import pytest

PARSERS = os.path.join("scripts", "protocolo_c", "revp_v1ud_source_specific_parsers.py")
RESOLVER = os.path.join("scripts", "protocolo_c", "revp_v1ud_source_url_resolver.py")
MANIFEST_SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ud_download_manifest_builder.py")
ACQ_SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ud_real_source_acquisition.py")
AUDIT_SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ud_raw_asset_integrity_audit.py")

SOURCES_CONFIG = os.path.join("configs", "protocolo_c", "ground_reference_evidence_sources.yaml")
TARGETS_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_real_acquisition_targets.yaml")
DOMAINS_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_allowed_domains.yaml")
POLICY_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_download_policy.yaml")
V1UC_EVIDENCE = os.path.join("datasets", "protocolo_c", "evidence_source_registry.csv")


@pytest.fixture
def full_dry_pipeline(tmp_path):
    out_dir = str(tmp_path / "datasets")
    local_dir = str(tmp_path / "local")

    res_out = os.path.join(out_dir, "v1ud_source_resolution_registry.csv")
    subprocess.run(
        [sys.executable, RESOLVER, "--sources-config", SOURCES_CONFIG,
         "--targets-config", TARGETS_CONFIG, "--domains-config", DOMAINS_CONFIG,
         "--out", res_out, "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )

    man_out = os.path.join(out_dir, "v1ud_download_manifest.csv")
    subprocess.run(
        [sys.executable, MANIFEST_SCRIPT, "--resolution-registry", res_out,
         "--policy-config", POLICY_CONFIG, "--domains-config", DOMAINS_CONFIG,
         "--local-only-dir", local_dir, "--out", man_out],
        capture_output=True, text=True, timeout=60,
    )

    subprocess.run(
        [sys.executable, ACQ_SCRIPT, "--manifest", man_out,
         "--out-dir", out_dir, "--local-only-dir", local_dir, "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )

    int_out = os.path.join(out_dir, "v1ud_raw_asset_integrity_registry.csv")
    delta_out = os.path.join(out_dir, "v1ud_gate_delta_registry.csv")
    subprocess.run(
        [sys.executable, AUDIT_SCRIPT,
         "--extraction-registry", os.path.join(out_dir, "v1ud_evidence_extraction_registry.csv"),
         "--v1uc-evidence", V1UC_EVIDENCE,
         "--out-integrity", int_out, "--out-gate-delta", delta_out],
        capture_output=True, text=True, timeout=60,
    )

    return out_dir


class TestParsersExecution:
    def test_parsers_run(self, full_dry_pipeline, tmp_path):
        out_dir = full_dry_pipeline
        actions_out = str(tmp_path / "actions.csv")
        report_out = str(tmp_path / "report.md")
        result = subprocess.run(
            [
                sys.executable, PARSERS,
                "--extraction-registry", os.path.join(out_dir, "v1ud_evidence_extraction_registry.csv"),
                "--resolution-registry", os.path.join(out_dir, "v1ud_source_resolution_registry.csv"),
                "--integrity-registry", os.path.join(out_dir, "v1ud_raw_asset_integrity_registry.csv"),
                "--gate-delta", os.path.join(out_dir, "v1ud_gate_delta_registry.csv"),
                "--sources-config", SOURCES_CONFIG,
                "--out-actions", actions_out,
                "--out-report", report_out,
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        assert os.path.exists(actions_out)
        assert os.path.exists(report_out)

    def test_actions_have_valid_types(self, full_dry_pipeline, tmp_path):
        out_dir = full_dry_pipeline
        actions_out = str(tmp_path / "actions.csv")
        report_out = str(tmp_path / "report.md")
        subprocess.run(
            [
                sys.executable, PARSERS,
                "--extraction-registry", os.path.join(out_dir, "v1ud_evidence_extraction_registry.csv"),
                "--resolution-registry", os.path.join(out_dir, "v1ud_source_resolution_registry.csv"),
                "--integrity-registry", os.path.join(out_dir, "v1ud_raw_asset_integrity_registry.csv"),
                "--gate-delta", os.path.join(out_dir, "v1ud_gate_delta_registry.csv"),
                "--sources-config", SOURCES_CONFIG,
                "--out-actions", actions_out,
                "--out-report", report_out,
            ],
            capture_output=True, text=True, timeout=60,
        )
        valid = {
            "DOWNLOAD_RETRY", "MANUAL_REVIEW", "FORMAL_REQUEST_REQUIRED",
            "LICENSE_REVIEW_REQUIRED", "PDF_TEXT_REVIEW", "GEOMETRY_AUDIT_REQUIRED",
            "PHENOMENON_SEPARATION_REQUIRED", "EVENT_SPECIFIC_SOURCE_NEEDED",
            "KEEP_CONTEXT_ONLY", "REJECT_AS_GROUND_REFERENCE", "METADATA_ONLY",
        }
        with open(actions_out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["action_type"] in valid, f"Invalid action: {row['action_type']}"

    def test_report_has_guardrails(self, full_dry_pipeline, tmp_path):
        out_dir = full_dry_pipeline
        actions_out = str(tmp_path / "actions.csv")
        report_out = str(tmp_path / "report.md")
        subprocess.run(
            [
                sys.executable, PARSERS,
                "--extraction-registry", os.path.join(out_dir, "v1ud_evidence_extraction_registry.csv"),
                "--resolution-registry", os.path.join(out_dir, "v1ud_source_resolution_registry.csv"),
                "--integrity-registry", os.path.join(out_dir, "v1ud_raw_asset_integrity_registry.csv"),
                "--gate-delta", os.path.join(out_dir, "v1ud_gate_delta_registry.csv"),
                "--sources-config", SOURCES_CONFIG,
                "--out-actions", actions_out,
                "--out-report", report_out,
            ],
            capture_output=True, text=True, timeout=60,
        )
        with open(report_out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ground_truth_operational" in content
        assert "can_create_training_label" in content
        assert "SUPPORT_ONLY" in content
        assert "no_coordinates_invented" in content

    def test_emdat_context_only(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ud_source_specific_parsers import parse_emdat
        result = parse_emdat({"url": "https://www.emdat.be/", "acquisition_status": "SKIPPED"})
        assert result["action"] == "KEEP_CONTEXT_ONLY"

    def test_charter_quickview_contextual(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ud_source_specific_parsers import parse_charter
        result = parse_charter({"url": "https://disasterscharter.org/activation-756", "acquisition_status": "SKIPPED"})
        assert result["quickview_only"] is True
        assert result["action"] == "KEEP_CONTEXT_ONLY"

    def test_sgb_formal_request(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ud_source_specific_parsers import parse_sgb_cprm
        result = parse_sgb_cprm({"url": "https://rigeo.sgb.gov.br/", "acquisition_status": "SKIPPED"})
        assert result["action"] == "FORMAL_REQUEST_REQUIRED"
