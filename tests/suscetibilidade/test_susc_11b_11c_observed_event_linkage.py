"""SUSC-11B/11C observed-event linkage tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
VALIDATOR = ROOT / "scripts" / "suscetibilidade" / "validate_susc_11b_11c_observed_event_linkage.py"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_11b_observed_event_catalog_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_11c_event_patch_linkage_v1.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11C_event_patch_linkage_summary.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11B_11C_observed_event_linkage_report.md"

MANDATORY = (
    "O SUSC-11B/11C organiza ocorrências reais/documentais de alagamento/inundação e "
    "testa sua ligação espacial com patches. Esses vínculos são evidência observacional "
    "review-only e não constituem ground truth supervisionado, predição operacional ou "
    "confirmação automática de ocorrência por patch."
)


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


def test_linkage_governance_closed():
    rows = read_csv(LINKAGE)
    assert rows
    assert all(r["can_be_ground_truth"] == "false" for r in rows)
    assert all(r["allowed_for_training"] == "false" for r in rows)
    assert all(r["review_only"] == "true" for r in rows)


def test_observed_evidence_requires_geometry_and_date():
    events = {r["event_id"]: r for r in read_csv(CATALOG)}
    for row in read_csv(LINKAGE):
        if row["can_be_used_as_observed_evidence"] != "true":
            continue
        event = events[row["event_id"]]
        assert event["bbox"] or (event["lat"] and event["lon"]) or event["geojson_ref"]
        assert event["event_date"] or event["event_period_start"]
        assert row["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"


def test_contextual_levels_are_not_strong():
    blocked = {
        "risk_area_context",
        "alert_only",
        "administrative_record_only",
        "documentary_context_only",
    }
    for row in read_csv(LINKAGE):
        if row["evidence_level"] in blocked:
            assert row["linkage_confidence"] != "strong"
            assert row["can_be_used_as_observed_evidence"] == "false"


def test_summary_metrics_and_report_statement():
    metrics = {r["metric"]: r["value"] for r in read_csv(SUMMARY)}
    for key in (
        "total_events",
        "total_patches",
        "total_links",
        "links_by_region",
        "links_by_spatial_relation",
        "links_by_evidence_level",
        "strong_links",
        "moderate_links",
        "weak_links",
        "events_linked_to_any_patch",
        "patches_with_observed_event_evidence",
        "patches_with_only_contextual_evidence",
        "review_only_count",
        "training_allowed_count",
        "ground_truth_count",
    ):
        assert key in metrics
    assert metrics["training_allowed_count"] == "0"
    assert metrics["ground_truth_count"] == "0"
    assert MANDATORY in REPORT.read_text(encoding="utf-8")
