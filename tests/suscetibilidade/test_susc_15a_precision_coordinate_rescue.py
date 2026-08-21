"""SUSC-15A precision coordinate rescue tests."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"

GEOCODED = DAT / "susc_15a_precision_geocoded_occurrences_v1.csv"
CATALOG = DAT / "susc_15a_calibration_eligible_event_catalog_v1.csv"
LINKAGE = DAT / "susc_15a_precision_event_patch_linkage_v1.csv"
REPORT = OUT / "SUSC_15A_PRECISION_observed_event_coordinate_rescue_report.md"
MANDATORY = (
    "O SUSC-15A-PRECISION tenta obter evidencia espacial precisa para eventos observados usando apenas geometrias "
    "diretas, referencias oficiais de enderecamento/intersecao/linear referencing e footprints rastreaveis. A etapa "
    "nao usa bairro como coordenada, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico."
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=ROOT, text=True,
                          capture_output=True, timeout=900, check=False)


def test_pipeline_and_validator_pass():
    for name in (
        "audit_susc_15a_official_occurrence_address_content.py",
        "discover_susc_15a_precision_reference_sources.py",
        "acquire_susc_15a_precision_reference_sources.py",
        "parse_susc_15a_precision_reference_sources.py",
        "geocode_susc_15a_occurrences_with_precision_references.py",
        "build_susc_15a_event_footprint_candidates.py",
        "build_susc_15a_calibration_eligible_event_catalog.py",
        "build_susc_15a_precision_event_patch_linkage.py",
        "run_susc_15a_precision_score_diagnostics.py",
        "validate_susc_15a_precision_coordinate_rescue.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_governance_closed_and_score_v7_absent():
    for path in (GEOCODED, CATALOG, LINKAGE):
        rows = read_csv(path)
        if path != CATALOG:
            assert rows
        assert all(row.get("allowed_for_training") in {"", "false"} for row in rows)
        assert all(row.get("can_be_ground_truth") in {"", "false"} for row in rows)
        assert all(row.get("can_be_used_as_ground_truth") in {"", "false"} for row in rows)
        assert all(row.get("review_only") == "true" for row in rows)
    assert not (DAT / "susc_score_v7_candidate_by_patch_v1.csv").exists()


def test_only_precise_tiers_are_calibration_eligible():
    allowed = {
        "T0_official_event_polygon",
        "T1_official_event_point",
        "T2_official_address_point_or_parcel",
        "T3_official_intersection_point",
        "T4_official_house_number_linear_reference",
    }
    for row in read_csv(GEOCODED):
        if row.get("precision_tier") in {"T5_official_street_segment_candidate", "T6_neighborhood_context_only", "T7_rejected_or_non_flood"}:
            assert row.get("calibration_eligible") == "false"
        if row.get("calibration_eligible") == "true":
            assert row.get("precision_tier") in allowed
            assert row.get("lat") and row.get("lon") or row.get("wkt") or row.get("bbox") or row.get("geojson_ref")
    for row in read_csv(CATALOG):
        assert row.get("precision_tier") in allowed
        assert row.get("calibration_eligible") == "true"


def test_observational_evaluation_requires_valid_relation():
    for row in read_csv(LINKAGE):
        if row.get("can_be_used_for_observational_evaluation") == "true":
            assert row.get("spatial_relation") not in {"insufficient_for_patch_link", "ambiguous_multiple_patches"}
            assert row.get("linkage_confidence") in {"strong", "moderate"}


def test_report_mandatory_statement():
    assert MANDATORY in REPORT.read_text(encoding="utf-8")
