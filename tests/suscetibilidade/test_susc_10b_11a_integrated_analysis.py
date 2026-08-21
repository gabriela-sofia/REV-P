"""SUSC-10B->11A integrated analysis tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_10B_11A_INTEGRATED_ANALYSIS_REPORT.md"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_10b_11a_integrated_manifest_v1.json"


def test_report_and_manifest_exist():
    assert REPORT.exists()
    assert MANIFEST.exists()


def test_report_mandatory_statement():
    assert "não cria ground truth, não valida ocorrência real por patch" in REPORT.read_text(encoding="utf-8")


def test_manifest_governance():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["can_be_used_as_ground_truth"] is False
    assert m["allowed_for_training"] is False
    assert m["review_only"] is True
    assert m["trained_model"] is False
    assert m["model_persisted"] is False


def test_manifest_phases():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for ph in ("10B_score_comparison", "10C_score_sensitivity", "10D_dino_diagnostics", "11A_public_visual_package"):
        assert ph in m["phases"]
