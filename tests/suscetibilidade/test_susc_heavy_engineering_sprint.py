"""
SUSC heavy engineering sprint master tests.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_HEAVY_ENGINEERING_SPRINT_09_10_REPORT.md"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_heavy_engineering_sprint_manifest_v1.json"

HARDENING = ["susc_io.py", "susc_governance.py", "susc_aliases.py", "susc_geometry.py",
             "susc_downloads.py", "susc_common.py"]


def test_hardening_modules_exist():
    for m in HARDENING:
        assert (SCRIPTS / m).exists(), m


def test_master_report_exists_and_mandatory():
    assert REPORT.exists()
    assert "Esta sprint avançou a infraestrutura operacional do REV-P" in REPORT.read_text(encoding="utf-8")


def test_master_manifest_governance():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["allowed_for_training"] is False
    assert m["can_be_used_as_ground_truth"] is False
    assert m["review_only"] is True
    assert m["trained_model"] is False
    assert m["model_persisted"] is False


def test_sprint_phases_present():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert "10A_score_v6_candidate" in m["phases"]
    assert "09A_human_review_workspace" in m["phases"]
