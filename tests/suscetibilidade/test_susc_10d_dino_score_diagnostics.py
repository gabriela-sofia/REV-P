"""SUSC-10D DINO score diagnostics tests (absence path tolerant)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_public" / "suscetibilidade"
DISCOVERY = OUT / "SUSC_10D_dino_embedding_discovery.csv"
DISCOVERY_SUMMARY = OUT / "SUSC_10D_dino_embedding_discovery_summary.json"
ALIGN = OUT / "SUSC_10D_dino_score_alignment.csv"
SUMMARY = OUT / "SUSC_10D_dino_score_diagnostics_summary.json"
REPORT = OUT / "SUSC_10D_dino_score_diagnostics_report.md"


def test_artifacts_exist():
    for p in (DISCOVERY, DISCOVERY_SUMMARY, ALIGN, SUMMARY, REPORT):
        assert p.exists(), p


def test_dino_usage_diagnostic_only():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["dino_usage"] == "representational_diagnostic_only_never_classifier"


def test_governance():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["can_be_used_as_ground_truth"] is False
    assert s["allowed_for_training"] is False
    assert s["trained_model"] is False
    assert s["model_persisted"] is False


def test_status_consistent():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert "dino_status" in s
    if s["dino_status"].startswith("ABSENT"):
        assert s["loadable_per_patch_vectors_available"] is False


def test_report_mandatory_statement():
    assert "não treina classificador" in REPORT.read_text(encoding="utf-8")
