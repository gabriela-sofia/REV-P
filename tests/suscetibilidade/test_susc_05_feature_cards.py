"""
SUSC-05 tests — feature cards, chain register, review-only invariants.

Assert the SUSC-05 artifacts exist, cover the obligatory features, preserve the
review-only invariants, never release an unresolved feature into score v6, and
register the three SUSC-01..04 commits. No score, model, or ground truth created.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

CARD_SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_feature_card_schema_v1.json"
FC_DIR = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards"
CARDS_JSON = FC_DIR / "susc_feature_cards_v1.json"
CATALOG_MD = FC_DIR / "SUSC_05_feature_cards_catalog.md"
SUMMARY_CSV = FC_DIR / "SUSC_05_feature_cards_summary.csv"
REPORT_MD = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_05_feature_cards_methodology_report.md"
)
CHAIN_REGISTER = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_01_04_chain_closure_register.md"
)

OBLIGATORY = [
    "elevation_mean", "slope_mean", "hand_mean", "twi_mean", "tpi_250m_mean",
    "curvature_laplacian_mean", "distance_to_water_mean", "water_occurrence_patch",
    "flow_acc_log_mean", "flow_acc_log_p75", "chirps_3d_mm", "chirps_7d_mm",
    "chirps_30d_mm", "rain_persistence_index", "s1_vv_mean_clean", "s1_vh_mean_clean",
    "s1_vv_minus_vh_mean_clean", "ndvi_mean", "mndwi_mean", "ndbi_mean",
    "urban_prop", "vegetation_prop",
    "chirps_3d_to_30d_ratio", "chirps_7d_to_30d_ratio", "runoff_score",
]

COMMIT_HASHES = ["4b49eb5", "7986da4", "3c49608"]


@pytest.fixture(scope="module")
def cards():
    with CARDS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)["cards"]


def test_susc05_artifacts_exist():
    for p in (CARD_SCHEMA, CARDS_JSON, CATALOG_MD, SUMMARY_CSV, REPORT_MD, CHAIN_REGISTER):
        assert p.exists(), f"missing artifact: {p}"


def test_feature_cards_json_valid(cards):
    assert isinstance(cards, list) and cards
    with CARD_SCHEMA.open("r", encoding="utf-8") as f:
        required = json.load(f)["required_fields"]
    for c in cards:
        for field in required:
            assert field in c, f"{c.get('feature_name')} missing {field}"


def test_obligatory_features_present(cards):
    names = {c["feature_name"] for c in cards}
    missing = [f for f in OBLIGATORY if f not in names]
    assert not missing, f"missing obligatory cards: {missing}"


def test_review_only_invariants(cards):
    for c in cards:
        assert c["can_be_ground_truth"] is False, c["feature_name"]
        assert c["allowed_for_training"] is False, c["feature_name"]
        assert c["review_only"] is True, c["feature_name"]


def test_no_ground_truth_created(cards):
    assert all(c["can_be_ground_truth"] is False for c in cards)


def test_no_training_unlocked(cards):
    assert all(c["allowed_for_training"] is False for c in cards)


def test_unresolved_not_in_score_v6(cards):
    for c in cards:
        if c["status"] == "unresolved":
            assert c["safe_for_score_v6"] is False, c["feature_name"]


def test_heuristics_are_proxy_only(cards):
    for c in cards:
        if c["feature_group"] in {"score", "heuristic_label", "proxy_v5"}:
            assert c["status"] == "proxy_only", c["feature_name"]
            assert c["can_be_ground_truth"] is False
            assert c["safe_for_score_v6"] is False


def test_report_has_mandatory_statement():
    text = REPORT_MD.read_text(encoding="utf-8")
    assert "não calcula score, não treina modelo e não cria ground truth" in text


def test_chain_register_contains_three_commits():
    reg = CHAIN_REGISTER.read_text(encoding="utf-8")
    for h in COMMIT_HASHES:
        assert h in reg, f"commit {h} not in chain register"
