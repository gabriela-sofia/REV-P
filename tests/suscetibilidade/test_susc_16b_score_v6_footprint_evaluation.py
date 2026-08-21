"""SUSC-16B footprint score v6 observational evaluation tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

DATASET = DAT / "susc_16b_footprint_event_control_dataset_v1.csv"
SUMMARY = OUT / "SUSC_16B_score_v6_footprint_evaluation_summary.json"
FEATURE_AUDIT = OUT / "SUSC_16B_feature_direction_audit.json"
QUALITY_SUMMARY = OUT / "SUSC_16B_footprint_evidence_quality_summary.json"
READINESS = OUT / "SUSC_16B_readiness_for_16C_and_score_v7.csv"
REPORT = OUT / "SUSC_16B_score_v6_footprint_observational_evaluation_report.md"
MANDATORY = (
    "O SUSC-16B avalia o score v6 contra footprints observacionais elegíveis "
    "em modo review-only. A etapa não cria ground truth, não libera treino "
    "supervisionado, não cria score v7 e trata controles como ausência "
    "documentada, não como negativos verdadeiros."
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_pipeline_and_validator_pass():
    for name in (
        "build_susc_16b_footprint_event_control_dataset.py",
        "run_susc_16b_score_v6_footprint_evaluation.py",
        "run_susc_16b_feature_contrast_against_footprints.py",
        "build_susc_16b_low_score_footprint_case_audit.py",
        "audit_susc_16b_footprint_evidence_quality.py",
        "build_susc_16b_proxy_calibration_recommendations.py",
        "run_susc_16b_readiness_for_16c_and_score_v7.py",
        "validate_susc_16b_score_v6_footprint_evaluation.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_dataset_governance_and_control_semantics():
    rows = read_csv(DATASET)
    assert rows
    assert all(row["allowed_for_training"] == "false" for row in rows)
    assert all(row["can_be_ground_truth"] == "false" for row in rows)
    assert all(row["review_only"] == "true" for row in rows)
    controls = [row for row in rows if row["group_type"] == "no_documented_footprint_control"]
    assert controls
    assert all(row["evidence_strength"] != "negative" for row in controls)
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_metrics_and_readiness_are_review_only():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["event_patch_count"] == 65
    assert summary["unique_event_patches"] == 62
    assert summary["score_v7_created"] is False
    readiness = read_csv(READINESS)
    assert {row["criterion"]: row["value"] for row in readiness}["ready_for_score_v7_candidate"] == "false"


def test_feature_and_quality_outputs_exist():
    feature_audit = json.loads(FEATURE_AUDIT.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY_SUMMARY.read_text(encoding="utf-8"))
    assert "features_agreeing" in feature_audit
    assert "features_diverging" in feature_audit
    assert quality["eligible_footprint_count"] >= 10
    assert quality["review_only"] is True


def test_report_mandatory_statement():
    assert MANDATORY in REPORT.read_text(encoding="utf-8")
