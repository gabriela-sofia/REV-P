"""
SUSC-04 tests — provenance audit, scientific direction, decision matrix.

Assert the SUSC-04 artifacts exist, cover the critical features, preserve the
review-only invariants, and never release an unresolved feature into score v6.
No model is trained; no ground truth is created.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

DIRECTION_PATH = (
    ROOT / "schemas" / "suscetibilidade" / "susc_feature_scientific_direction_v1.json"
)
AUDIT_CSV = ROOT / "manifests" / "suscetibilidade" / "susc_feature_provenance_audit_v1.csv"
DECISION_CSV = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_04_score_v6_feature_decision_matrix.csv"
)
REPORT_PATH = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_04_feature_provenance_audit_report.md"
)

CRITICAL_FEATURES = [
    "elevation_mean", "slope_mean", "hand_mean", "twi_mean", "tpi_250m_mean",
    "curvature_laplacian_mean", "distance_to_water_mean", "water_occurrence_patch",
    "flow_acc_log_mean", "flow_acc_log_p75", "chirps_3d_mm", "chirps_7d_mm",
    "chirps_30d_mm", "rain_persistence_index", "s1_vv_mean_clean", "s1_vh_mean_clean",
    "ndvi_mean", "mndwi_mean", "ndbi_mean", "urban_prop", "vegetation_prop",
    "urban_water_interaction", "urban_drainage_interaction",
]


@pytest.fixture(scope="module")
def audit_rows():
    with AUDIT_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def decision_rows():
    with DECISION_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_direction_schema_exists_and_valid():
    assert DIRECTION_PATH.exists()
    with DIRECTION_PATH.open("r", encoding="utf-8") as f:
        d = json.load(f)
    assert d["features"], "direction vocabulary must not be empty"
    allowed = set(d["direction_vocabulary"])
    for feat in d["features"]:
        assert feat["expected_direction_for_flood_susceptibility"] in allowed


def test_audit_csv_exists(audit_rows):
    assert AUDIT_CSV.exists()
    assert len(audit_rows) >= len(CRITICAL_FEATURES)


def test_decision_matrix_exists(decision_rows):
    assert DECISION_CSV.exists()
    assert decision_rows


def test_report_exists():
    assert REPORT_PATH.exists()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "não cria ground truth" in text
    assert "não desbloqueia treinamento supervisionado" in text


def test_critical_features_present(audit_rows):
    names = {r["feature_name"] for r in audit_rows}
    missing = [f for f in CRITICAL_FEATURES if f not in names]
    assert not missing, f"missing critical features: {missing}"


def test_review_only_invariants(audit_rows):
    for r in audit_rows:
        assert r["can_be_ground_truth"] == "false", r["feature_name"]
        assert r["allowed_for_training"] == "false", r["feature_name"]
        assert r["review_only"] == "true", r["feature_name"]


def test_unresolved_not_in_score_v6(audit_rows):
    for r in audit_rows:
        if r["provenance_status"] == "unresolved":
            assert r["safe_for_score_v6"] == "false", r["feature_name"]


def test_ground_truth_stays_false(audit_rows):
    assert all(r["can_be_ground_truth"] == "false" for r in audit_rows)


def test_allowed_for_training_stays_false(audit_rows):
    assert all(r["allowed_for_training"] == "false" for r in audit_rows)


def test_no_ambiguous_direction_with_high_weight(decision_rows):
    for r in decision_rows:
        if r["direction"] in {"ambiguous", "non_monotonic"}:
            assert r["weight_class"] != "high", r["feature_name"]


def test_absent_requested_names_are_blocked(audit_rows):
    by_name = {r["feature_name"]: r for r in audit_rows}
    for name in ("chirps_3d_to_30d_ratio", "chirps_7d_to_30d_ratio", "runoff_score"):
        assert name in by_name
        assert by_name[name]["exists_in_susc03"] == "false"
        assert by_name[name]["safe_for_score_v6"] == "false"
