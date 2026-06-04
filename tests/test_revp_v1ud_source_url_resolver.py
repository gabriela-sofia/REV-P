"""Tests for v1ud — Source URL Resolver."""

import csv
import os
import subprocess
import sys

import pytest

RESOLVER = os.path.join("scripts", "protocolo_c", "revp_v1ud_source_url_resolver.py")
SOURCES_CONFIG = os.path.join("configs", "protocolo_c", "ground_reference_evidence_sources.yaml")
TARGETS_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_real_acquisition_targets.yaml")
DOMAINS_CONFIG = os.path.join("configs", "protocolo_c", "v1ud_allowed_domains.yaml")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class TestURLResolver:
    def test_dry_run_produces_registry(self, tmp_path):
        out = str(tmp_path / "res.csv")
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
        assert os.path.exists(out)

    def test_required_columns(self, tmp_path):
        out = str(tmp_path / "res.csv")
        subprocess.run(
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
        required = [
            "resolution_id", "source_id", "event_id", "base_url",
            "candidate_url", "http_status", "acquisition_decision",
        ]
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in required:
            assert col in cols

    def test_dry_run_all_dry_run_status(self, tmp_path):
        out = str(tmp_path / "res.csv")
        subprocess.run(
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
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["http_status"] == "DRY_RUN"


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
class TestDomainConfig:
    def test_all_allowed_domains_have_category(self):
        with open(DOMAINS_CONFIG, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for domain in config.get("allowed_domains", []):
            assert "category" in domain
            assert "domain" in domain

    def test_targets_reference_known_sources(self):
        with open(SOURCES_CONFIG, "r", encoding="utf-8") as f:
            sources = yaml.safe_load(f)
        source_ids = {s["source_id"] for s in sources.get("sources", [])}

        with open(TARGETS_CONFIG, "r", encoding="utf-8") as f:
            targets = yaml.safe_load(f)
        for key in ["priority_1", "priority_2", "priority_3"]:
            for block in targets.get(key, []):
                assert block["source_id"] in source_ids, f"{block['source_id']} not in sources"


class TestDomainCheck:
    def test_check_domain_function(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ud_source_url_resolver import check_domain, build_domain_index

        index = build_domain_index({
            "allowed_domains": [
                {"domain": "portal.inmet.gov.br", "category": "METEO", "download_allowed": True},
            ]
        })
        allowed, host, info = check_domain("https://portal.inmet.gov.br/dadoshistoricos", index)
        assert allowed is True
        assert info["category"] == "METEO"

    def test_unknown_domain_blocked(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ud_source_url_resolver import check_domain, build_domain_index

        index = build_domain_index({
            "allowed_domains": [
                {"domain": "portal.inmet.gov.br", "category": "METEO"},
            ]
        })
        allowed, host, info = check_domain("https://evil.example.com/data", index)
        assert allowed is False
