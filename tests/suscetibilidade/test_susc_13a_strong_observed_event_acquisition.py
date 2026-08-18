"""SUSC-13A strong observed-event acquisition tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "suscetibilidade" / "validate_susc_13a_strong_observed_event_acquisition.py"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_observed_events_parsed_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_event_patch_linkage_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_observed_event_acquisition_report.md"

MANDATORY = (
    "O SUSC-13A busca evidências observacionais mais fortes de alagamento/inundação "
    "para reduzir a fragilidade detectada em SUSC-11/12. Mesmo eventos fortes permanecem "
    "review-only nesta etapa e não criam ground truth, treino supervisionado ou confirmação "
    "operacional automática por patch."
)

STRONG_LEVELS = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
WEAK_NOT_OBSERVED = {"weak_risk_area_context", "weak_alert_context", "weak_administrative_context"}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_validator_passes():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_parsed_governance_and_strong_constraints():
    rows = read_csv(PARSED)
    assert rows
    assert all(r["can_be_ground_truth"] == "false" for r in rows)
    assert all(r["allowed_for_training"] == "false" for r in rows)
    assert all(r["review_only"] == "true" for r in rows)
    for row in rows:
        if row["evidence_level"] in STRONG_LEVELS:
            assert row["bbox"] or (row["lat"] and row["lon"]) or row["wkt"] or row["geojson_ref"]
            assert row["event_date"] or row["event_period_start"]


def test_risk_alert_admin_not_observed():
    for row in read_csv(PARSED):
        if row["evidence_level"] in WEAK_NOT_OBSERVED:
            assert row["is_observed_event"] == "false"
        text = " ".join([row["source_name"], row["notes"], row["evidence_level"]]).lower()
        if "risk" in text or "risco" in text or "alerta" in text or "alert" in text:
            assert row["evidence_level"] not in STRONG_LEVELS


def test_linkage_review_only_and_no_gt_training():
    rows = read_csv(LINKAGE)
    assert rows
    assert all(r["can_be_ground_truth"] == "false" for r in rows)
    assert all(r["allowed_for_training"] == "false" for r in rows)
    assert all(r["review_only"] == "true" for r in rows)
    for row in rows:
        if row["can_be_used_for_observational_evaluation"] == "true":
            assert row["spatial_relation"] not in {"same_region_period_context", "insufficient_for_patch_link"}


def test_report_mandatory_statement_and_score_v7_absent():
    assert MANDATORY in REPORT.read_text(encoding="utf-8")
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
