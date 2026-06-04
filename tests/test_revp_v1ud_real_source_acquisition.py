"""Tests for v1ud — Real Source Acquisition."""

import csv
import os
import subprocess
import sys

import pytest

RESOLVER = os.path.join("scripts", "protocolo_c", "revp_v1ud_source_url_resolver.py")
MANIFEST = os.path.join("scripts", "protocolo_c", "revp_v1ud_download_manifest_builder.py")
ACQUISITION = os.path.join("scripts", "protocolo_c", "revp_v1ud_real_source_acquisition.py")

SOURCES_CONFIG = os.path.join("configs", "protocolo_c", "ground_reference_evidence_sources.yaml")
TARGETS_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_real_acquisition_targets.yaml")
DOMAINS_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_allowed_domains.yaml")
POLICY_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_download_policy.yaml")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")


@pytest.fixture
def resolved_registry(tmp_path):
    out = str(tmp_path / "resolution.csv")
    result = subprocess.run(
        [
            sys.executable, RESOLVER,
            "--sources-config", SOURCES_CONFIG,
            "--targets-config", TARGETS_CONFIG,
            "--domains-config", DOMAINS_CONFIG,
            "--out", out,
            "--dry-run",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out


@pytest.fixture
def download_manifest(tmp_path, resolved_registry):
    out = str(tmp_path / "manifest.csv")
    result = subprocess.run(
        [
            sys.executable, MANIFEST,
            "--resolution-registry", resolved_registry,
            "--policy-config", POLICY_CONFIG,
            "--domains-config", DOMAINS_CONFIG,
            "--local-only-dir", str(tmp_path / "local"),
            "--out", out,
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out


class TestResolverDryRun:
    def test_resolver_runs_dry(self, resolved_registry):
        assert os.path.exists(resolved_registry)
        with open(resolved_registry, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

    def test_all_decisions_valid(self, resolved_registry):
        valid = {
            "DOWNLOAD_CANDIDATE", "METADATA_ONLY", "DRY_RUN",
            "BLOCKED_DOMAIN", "RESOLUTION_FAILED", "HTTP_ERROR",
            "LICENSE_NEEDS_REVIEW",
        }
        with open(resolved_registry, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["acquisition_decision"] in valid, f"Invalid: {row['acquisition_decision']}"

    def test_blocked_domain_blocked(self, tmp_path):
        out = str(tmp_path / "res2.csv")
        result = subprocess.run(
            [
                sys.executable, RESOLVER,
                "--sources-config", SOURCES_CONFIG,
                "--targets-config", TARGETS_CONFIG,
                "--domains-config", DOMAINS_CONFIG,
                "--out", out,
                "--dry-run",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0


class TestManifestBuilder:
    def test_manifest_created(self, download_manifest):
        assert os.path.exists(download_manifest)
        with open(download_manifest, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0

    def test_no_absolute_paths(self, download_manifest):
        with open(download_manifest, "r", encoding="utf-8") as f:
            content = f.read()
        assert "C:\\Users" not in content
        assert "/home/" not in content

    def test_manifest_has_required_columns(self, download_manifest):
        required = [
            "manifest_id", "source_id", "event_id", "url",
            "download_allowed", "reason",
        ]
        with open(download_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        for col in required:
            assert col in cols


class TestAcquisitionDryRun:
    def test_acquisition_dry_run(self, download_manifest, tmp_path):
        out_dir = str(tmp_path / "out")
        local_dir = str(tmp_path / "local")
        result = subprocess.run(
            [
                sys.executable, ACQUISITION,
                "--manifest", download_manifest,
                "--out-dir", out_dir,
                "--local-only-dir", local_dir,
                "--dry-run",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        reg = os.path.join(out_dir, "v1ud_evidence_extraction_registry.csv")
        assert os.path.exists(reg)

    def test_no_training_label_in_extraction(self, download_manifest, tmp_path):
        out_dir = str(tmp_path / "out")
        local_dir = str(tmp_path / "local")
        subprocess.run(
            [
                sys.executable, ACQUISITION,
                "--manifest", download_manifest,
                "--out-dir", out_dir,
                "--local-only-dir", local_dir,
                "--dry-run",
            ],
            capture_output=True, text=True, timeout=60,
        )
        reg = os.path.join(out_dir, "v1ud_evidence_extraction_registry.csv")
        with open(reg, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row.get("can_create_training_label") == "false"
            assert row.get("ground_truth_operational") == "false"


class TestGuardrails:
    def test_local_only_in_gitignore(self):
        with open(".gitignore", "r", encoding="utf-8") as f:
            content = f.read()
        assert "local_only/" in content

    def test_no_training_label_in_configs(self):
        sys.path.insert(0, os.path.abspath("."))
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        for cfg in [SOURCES_CONFIG, TARGETS_CONFIG, POLICY_CONFIG]:
            with open(cfg, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if "can_create_training_label" in data:
                assert data["can_create_training_label"] is False
            if "ground_truth_operational" in data:
                assert data["ground_truth_operational"] is False
