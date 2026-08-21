"""SUSC-13B-AUTO automatic discovery/acquisition tests (offline-safe)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DATASETS = ROOT / "datasets" / "suscetibilidade"
MANIFESTS = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

VALIDATOR = SCRIPTS / "validate_susc_13b_auto_official_event_acquisition.py"
PARSED = DATASETS / "susc_13b_auto_observed_events_parsed_v1.csv"
CATALOG = DATASETS / "susc_13b_auto_consolidated_observed_event_catalog_v1.csv"
LINKAGE = DATASETS / "susc_13b_auto_event_patch_linkage_v1.csv"
DISCOVERED = MANIFESTS / "susc_13b_auto_discovered_sources_v1.csv"
QUERY_MANIFEST = MANIFESTS / "susc_13b_auto_discovery_query_manifest_v1.csv"
READINESS = OUT / "SUSC_13B_auto_observational_readiness.csv"
STRONG_REPORT = OUT / "SUSC_13B_AUTO_official_event_discovery_acquisition_report.md"

MANDATORY = (
    "O SUSC-13B-AUTO realiza descoberta e aquisição automática de fontes oficiais/rastreáveis "
    "para fortalecer a camada observacional de alagamento/inundação. Mesmo quando encontra "
    "eventos fortes, a etapa mantém todos os vínculos em modo review-only, não cria ground truth, "
    "não treina modelo supervisionado e não cria score v7 automaticamente."
)
STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
WEAK_NOT_OBSERVED = {"weak_risk_area_context", "weak_alert_context", "weak_administrative_context"}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run(script: Path):
    return subprocess.run([sys.executable, str(script)], cwd=ROOT, text=True,
                          capture_output=True, timeout=180, check=False)


def test_pipeline_runs_offline_then_validator_passes():
    # Full offline pipeline must run and validate without network.
    for name in (
        "discover_susc_13b_official_event_sources.py",
        "acquire_susc_13b_auto_discovered_sources.py",
        "parse_susc_13b_auto_observed_events.py",
        "build_susc_13b_auto_consolidated_event_catalog.py",
        "build_susc_13b_auto_event_patch_linkage.py",
        "run_susc_13b_auto_observational_readiness.py",
        "run_susc_13b_auto_event_score_diagnostics.py",
    ):
        r = _run(SCRIPTS / name)
        assert r.returncode == 0, name + "\n" + r.stdout + r.stderr
    r = _run(VALIDATOR)
    assert r.returncode == 0, r.stdout + r.stderr


def test_discovery_query_plan_and_sources_exist():
    queries = read_csv(QUERY_MANIFEST)
    sources = read_csv(DISCOVERED)
    assert queries and sources
    regions = {q["region"] for q in queries}
    assert {"recife", "petropolis", "curitiba"} <= regions
    assert all(s["review_only"] == "true" for s in sources)
    # no invented direct download: endpoint roots are not download candidates
    methods = {s["discovery_method"] for s in sources}
    assert {"ckan_api", "arcgis_rest"} <= methods


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


def test_report_mandatory_statement_and_score_v7_absent():
    assert MANDATORY in STRONG_REPORT.read_text(encoding="utf-8")
    assert not (DATASETS / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_readiness_does_not_unlock_training():
    readiness = {r["metric"]: r["value"] for r in read_csv(READINESS)}
    assert "ready_for_score_v7" in readiness
    # offline default: nothing is ready, but even when ready, score v7 is not created here.
    assert not (DATASETS / "susc_score_v7_candidate_by_patch_v1.csv").exists()
