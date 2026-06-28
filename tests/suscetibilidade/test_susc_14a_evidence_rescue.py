"""SUSC-14A evidence rescue tests.

Re-runs deterministic 14A steps over already available artifacts and checks the
fail-closed governance contract. It does not use generic geocoding, API keys, or
create score v7.
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

VALIDATOR = SCRIPTS / "validate_susc_14a_evidence_rescue.py"
GEOCODED = DATASETS / "susc_14a_geocoded_official_occurrences_v1.csv"
CATALOG = DATASETS / "susc_14a_rescued_observed_event_catalog_v1.csv"
LINKAGE = DATASETS / "susc_14a_event_patch_linkage_v1.csv"
READINESS = OUT / "SUSC_14A_observational_readiness.csv"
FINAL_REPORT = OUT / "SUSC_14A_evidence_rescue_report.md"

MANDATORY = (
    "O SUSC-14A tenta resgatar evidencia observacional a partir de registros oficiais sem coordenada "
    "explicita, usando apenas referencias espaciais oficiais/rastreaveis e footprints publicos quando "
    "disponiveis. Eventos geocodificados por referencia oficial permanecem review-only, nao criam "
    "ground truth, nao liberam treino supervisionado e nao autorizam score v7 automatico."
)
STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
OBSERVED = STRONG | {
    "moderate_official_occurrence_point",
    "moderate_official_occurrence_geocoded",
    "moderate_official_street_segment_match",
    "moderate_official_flood_bbox",
}
NON_PATCH_RELATIONS = {"same_neighborhood_context", "same_region_period_context", "insufficient_for_patch_link"}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )


def test_deterministic_14a_pipeline_and_validator_pass():
    for name in (
        "build_susc_14a_occurrence_address_normalizer.py",
        "geocode_susc_14a_official_occurrences.py",
        "build_susc_14a_rescued_observed_event_catalog.py",
        "build_susc_14a_event_patch_linkage.py",
        "run_susc_14a_observational_diagnostics.py",
        "validate_susc_14a_evidence_rescue.py",
    ):
        result = run_script(name)
        assert result.returncode == 0, name + "\n" + result.stdout + result.stderr


def test_governance_flags_closed():
    for path in (GEOCODED, CATALOG, LINKAGE):
        rows = read_csv(path)
        assert rows
        assert all(row.get("allowed_for_training") == "false" for row in rows)
        assert all(row.get("can_be_ground_truth") == "false" for row in rows)
        assert all(row.get("review_only") == "true" for row in rows)
        if path != LINKAGE:
            assert all(row.get("can_be_used_as_ground_truth") == "false" for row in rows)
    assert all(row.get("can_be_used_as_ground_truth") == "false" for row in read_csv(LINKAGE))


def test_no_score_v7_ground_truth_training_or_model_artifact():
    assert not (DATASETS / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    model_exts = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}
    for base in (DATASETS, OUT, SCRIPTS):
        assert not [path for path in base.rglob("*") if path.suffix.lower() in model_exts]


def test_strong_moderate_and_context_constraints():
    for row in read_csv(CATALOG):
        if row.get("evidence_level") in STRONG:
            assert row.get("event_date") or row.get("event_period_start")
            assert row.get("bbox") or (row.get("lat") and row.get("lon")) or row.get("wkt") or row.get("geojson_ref")

    for row in read_csv(GEOCODED):
        if row.get("evidence_level") in OBSERVED:
            assert row.get("georeferencing_status", "").startswith("official_")
        text = " ".join(
            row.get(col, "") for col in ("evidence_level", "georeferencing_status", "match_method", "matched_feature_type")
        )
        assert not ("street_segment" in text and row.get("evidence_level") in STRONG)

    for row in read_csv(LINKAGE):
        if row.get("spatial_relation") in NON_PATCH_RELATIONS:
            assert row.get("patch_id") in ("", "REGION_LEVEL_NO_PATCH_RESOLUTION")
            assert row.get("can_be_used_for_observational_evaluation") == "false"
        if row.get("can_be_used_for_observational_evaluation") == "true":
            assert row.get("evidence_level") in OBSERVED
            assert row.get("spatial_relation") not in NON_PATCH_RELATIONS


def test_readiness_false_when_no_strong_or_moderate_links():
    linkage = read_csv(LINKAGE)
    readiness = {row["metric"]: row["value"] for row in read_csv(READINESS)}
    strong_mod = [row for row in linkage if row.get("linkage_confidence") in {"strong", "moderate"}]
    if not strong_mod:
        assert readiness["ready_for_12a_temporal"] == "false"
        assert readiness["ready_for_12b_feature_contrast"] == "false"
        assert readiness["ready_for_12c_proxy_calibration"] == "false"
        assert readiness["ready_for_score_v7"] == "false"
        assert readiness["score_v7_status"] == "blocked"
        assert readiness["reason"] == "no_strong_or_moderate_patch_links"


def test_final_report_mandatory_statement_and_blocker_sections():
    report = FINAL_REPORT.read_text(encoding="utf-8")
    assert MANDATORY in report
    assert "## Achado negativo principal" in report
    assert "## Consequencia para score v7" in report
