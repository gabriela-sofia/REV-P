"""Tests for v1ui — Public Artifact Downloader."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_public_artifact_downloader.py")
DOMAINS = os.path.join("configs", "protocolo_c", "v1ui_allowed_domains.yaml")


def _make_manifest(tmp_path, entries):
    path = os.path.join(tmp_path, "manifest.csv")
    cols = ["crawl_id", "event_id", "source_id", "discovered_url",
            "extension", "artifact_candidate_status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(entries)
    return path


def _run(tmp_path, entries, extra_args=None):
    manifest = _make_manifest(tmp_path, entries)
    out = os.path.join(tmp_path, "downloads.csv")
    cmd = [sys.executable, SCRIPT, "--manifest", manifest,
           "--allowed-domains", DOMAINS, "--out", out,
           "--local-only-dir", os.path.join(tmp_path, "raw")]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPublicArtifactDownloader:
    def test_empty_manifest(self, tmp_path):
        rows = _run(str(tmp_path), [])
        assert len(rows) == 0

    def test_blocked_domain_not_downloaded(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "crawl_id": "C1", "event_id": "E1", "source_id": "S1",
            "discovered_url": "https://www.google.com/data.csv",
            "extension": ".csv", "artifact_candidate_status": "CANDIDATE",
        }])
        assert len(rows) == 1
        assert rows[0]["download_status"] == "BLOCKED_DOMAIN"

    def test_allowed_domain_dry_run(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "crawl_id": "C1", "event_id": "E1", "source_id": "S1",
            "discovered_url": "https://dados.recife.pe.gov.br/data.csv",
            "extension": ".csv", "artifact_candidate_status": "CANDIDATE",
        }])
        assert len(rows) == 1
        assert rows[0]["download_status"] == "DRY_RUN"

    def test_blocked_extension(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "crawl_id": "C1", "event_id": "E1", "source_id": "S1",
            "discovered_url": "https://dados.gov.br/evil.exe",
            "extension": ".exe", "artifact_candidate_status": "CANDIDATE",
        }])
        assert len(rows) == 1
        assert rows[0]["download_status"] == "BLOCKED_EXTENSION"

    def test_artifact_ids_unique(self, tmp_path):
        rows = _run(str(tmp_path), [
            {"crawl_id": "C1", "event_id": "E1", "source_id": "S1",
             "discovered_url": "https://dados.gov.br/a.csv",
             "extension": ".csv", "artifact_candidate_status": "CANDIDATE"},
            {"crawl_id": "C2", "event_id": "E2", "source_id": "S2",
             "discovered_url": "https://dados.gov.br/b.csv",
             "extension": ".csv", "artifact_candidate_status": "CANDIDATE"},
        ])
        ids = [r["artifact_id"] for r in rows]
        assert len(ids) == len(set(ids))
