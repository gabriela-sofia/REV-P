"""Tests for v1ui — Public Source Discovery."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_public_source_discovery.py")
CONFIG = os.path.join("configs", "protocolo_c", "v1ui_public_source_targets.yaml")
EVENTS = os.path.join("datasets", "protocolo_c", "event_candidate_registry.csv")
DOMAINS = os.path.join("configs", "protocolo_c", "v1ui_allowed_domains.yaml")


def _run(tmp_path):
    out_dir = os.path.join(tmp_path, "out")
    os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run(
        [sys.executable, SCRIPT, "--config", CONFIG, "--events", EVENTS,
         "--allowed-domains", DOMAINS, "--out-dir", out_dir],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out_dir


class TestPublicSourceDiscovery:
    def test_runs_and_produces_output(self, tmp_path):
        out_dir = _run(str(tmp_path))
        assert os.path.exists(os.path.join(out_dir, "v1ui_public_discovery_registry.csv"))
        assert os.path.exists(os.path.join(out_dir, "v1ui_public_source_target_registry.csv"))

    def test_discovery_has_entries(self, tmp_path):
        out_dir = _run(str(tmp_path))
        with open(os.path.join(out_dir, "v1ui_public_discovery_registry.csv"), "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 3

    def test_all_events_covered(self, tmp_path):
        out_dir = _run(str(tmp_path))
        with open(os.path.join(out_dir, "v1ui_public_discovery_registry.csv"), "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        event_ids = {r["event_id"] for r in rows}
        assert "PET_2022_02_15" in event_ids
        assert "REC_2022_05_24_30" in event_ids

    def test_dry_run_method(self, tmp_path):
        out_dir = _run(str(tmp_path))
        with open(os.path.join(out_dir, "v1ui_public_discovery_registry.csv"), "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["discovery_method"] == "DRY_RUN"

    def test_discovery_ids_unique(self, tmp_path):
        out_dir = _run(str(tmp_path))
        with open(os.path.join(out_dir, "v1ui_public_discovery_registry.csv"), "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [r["discovery_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_candidate_classes_valid(self, tmp_path):
        out_dir = _run(str(tmp_path))
        valid = {"EVENT_SPECIFIC_PUBLIC_CANDIDATE", "CITY_PORTAL_PUBLIC_CANDIDATE",
                 "OPEN_DATA_PORTAL_CANDIDATE", "ARCGIS_REST_CANDIDATE",
                 "GEOSERVER_WFS_CANDIDATE", "DOCUMENT_REPOSITORY_CANDIDATE",
                 "GENERIC_HOMEPAGE", "NOT_RELEVANT"}
        with open(os.path.join(out_dir, "v1ui_public_discovery_registry.csv"), "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            assert r["candidate_class"] in valid, f"Invalid class: {r['candidate_class']}"
