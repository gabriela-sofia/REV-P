"""SUSC-13C-LIVE official event acquisition tests.

Does NOT re-run the live network probes (slow/network-dependent). It re-runs the
deterministic offline processing steps over the already-acquired artifacts and
asserts governance, classification and the validator.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DATASETS = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

VALIDATOR = SCRIPTS / "validate_susc_13c_live_official_event_acquisition.py"
HEALTH_JSON = OUT / "SUSC_13C_live_network_healthcheck.json"
PARSED = DATASETS / "susc_13c_live_observed_events_parsed_v1.csv"
CATALOG = DATASETS / "susc_13c_consolidated_observed_event_catalog_v1.csv"
LINKAGE = DATASETS / "susc_13c_event_patch_linkage_v1.csv"
READINESS = OUT / "SUSC_13C_observational_readiness.csv"
FINAL_REPORT = OUT / "SUSC_13C_LIVE_official_event_acquisition_report.md"

MANDATORY = (
    "O SUSC-13C-LIVE executa aquisição online real de fontes oficiais/rastreáveis para tentar "
    "materializar eventos observados de alagamento/inundação com data e geometria. Mesmo quando "
    "eventos fortes ou moderados são encontrados, todos os vínculos permanecem review-only, sem "
    "ground truth, sem treino supervisionado, sem score v7 automático e sem uso operacional preditivo."
)
STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
WEAK_NOT_OBSERVED = {"weak_risk_area_context", "weak_alert_context", "weak_administrative_context"}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run(script: Path):
    return subprocess.run([sys.executable, str(script)], cwd=ROOT, text=True,
                          capture_output=True, timeout=240, check=False)


def test_deterministic_pipeline_reruns_and_validator_passes():
    # offline-deterministic steps over already-acquired artifacts
    for name in (
        "parse_susc_13c_live_observed_events.py",
        "build_susc_13c_consolidated_observed_event_catalog.py",
        "build_susc_13c_event_patch_linkage.py",
        "run_susc_13c_event_score_diagnostics.py",
    ):
        r = _run(SCRIPTS / name)
        assert r.returncode == 0, name + "\n" + r.stdout + r.stderr
    r = _run(VALIDATOR)
    assert r.returncode == 0, r.stdout + r.stderr


def test_healthcheck_recorded_real_attempts():
    import json
    health = json.loads(HEALTH_JSON.read_text(encoding="utf-8"))
    assert health.get("results"), "healthcheck must record per-domain results"
    for r in health["results"]:
        assert r.get("status_code") != "" or r.get("error"), "each domain has a status or error"


def test_parsed_and_catalog_governance_and_strong_constraints():
    for path in (PARSED, CATALOG):
        rows = read_csv(path)
        assert all(r.get("can_be_ground_truth") == "false" for r in rows)
        assert all(r.get("allowed_for_training") == "false" for r in rows)
        assert all(r.get("review_only") == "true" for r in rows)
        for row in rows:
            if row.get("evidence_level") in STRONG:
                assert row.get("bbox") or (row.get("lat") and row.get("lon")) or row.get("wkt") or row.get("geojson_ref")
                assert row.get("event_date") or row.get("event_period_start")


def test_risk_alert_admin_not_observed_or_patch_level():
    for row in read_csv(CATALOG):
        if row.get("evidence_level") in WEAK_NOT_OBSERVED:
            assert row.get("is_observed_event") in (None, "", "false")
    for row in read_csv(LINKAGE):
        if row.get("evidence_level") in WEAK_NOT_OBSERVED:
            assert row.get("patch_id") in ("", "REGION_LEVEL_NO_PATCH_RESOLUTION")


def test_linkage_review_only_and_no_gt_training():
    rows = read_csv(LINKAGE)
    assert rows
    assert all(r["can_be_ground_truth"] == "false" for r in rows)
    assert all(r["allowed_for_training"] == "false" for r in rows)
    assert all(r["review_only"] == "true" for r in rows)
    for row in rows:
        if row["can_be_used_for_observational_evaluation"] == "true":
            assert row["spatial_relation"] not in {"same_region_period_context", "insufficient_for_patch_link"}


def test_final_report_mandatory_and_score_v7_absent():
    assert MANDATORY in FINAL_REPORT.read_text(encoding="utf-8")
    assert not (DATASETS / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    readiness = {r["metric"]: r["value"] for r in read_csv(READINESS)}
    assert "ready_for_score_v7" in readiness
