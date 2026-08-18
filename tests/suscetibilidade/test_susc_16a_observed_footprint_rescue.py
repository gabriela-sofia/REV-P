"""SUSC-16A observed footprint rescue tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

CATALOG = DAT / "susc_16a_observed_footprint_catalog_v1.csv"
LINKAGE = DAT / "susc_16a_footprint_patch_linkage_v1.csv"
REPORT = OUT / "SUSC_16A_observed_footprint_rescue_report.md"
MANDATORY = (
    "O SUSC-16A substitui a tentativa de geocodificacao textual por uma "
    "estrategia de footprints observacionais, combinando geometrias locais, "
    "fontes oficiais/tecnicas e planejamento Sentinel/SAR. A etapa mantem "
    "todos os vinculos review-only, nao cria ground truth, nao libera treino "
    "supervisionado e nao cria score v7 automatico."
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
        "audit_susc_16a_local_project_event_geometry_sources.py",
        "parse_susc_16a_local_event_geometry_sources.py",
        "discover_susc_16a_external_flood_footprint_sources.py",
        "acquire_susc_16a_external_flood_footprint_sources.py",
        "build_susc_16a_sentinel_event_window_plan.py",
        "build_susc_16a_sentinel1_flood_mapping_targets.py",
        "run_susc_16a_local_sar_flood_canary_if_available.py",
        "build_susc_16a_observed_footprint_catalog.py",
        "build_susc_16a_footprint_patch_linkage.py",
        "run_susc_16a_footprint_score_diagnostics.py",
        "validate_susc_16a_observed_footprint_rescue.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_governance_closed_and_score_v7_absent():
    for path in (CATALOG, LINKAGE):
        rows = read_csv(path)
        assert rows
        assert all(row.get("allowed_for_training") == "false" for row in rows)
        assert all(row.get("can_be_ground_truth") == "false" for row in rows)
        assert all(row.get("review_only") == "true" for row in rows)
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_context_is_not_observed_eligible():
    for row in read_csv(CATALOG):
        if row.get("footprint_type") in {"official_risk_context", "official_susceptibility_reference"}:
            assert row.get("eligible_for_patch_observational_evaluation") == "false"


def test_report_mandatory_statement():
    assert MANDATORY in REPORT.read_text(encoding="utf-8")
