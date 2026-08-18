"""
SUSC-06A tests — proxy baseline, review-only invariants, no ground truth/model.

Assert the SUSC-06A artifacts exist, the target is a proxy (never ground truth),
no model file was persisted, and the mandatory statement is present. The baseline
is a methodological diagnostic against an internal proxy, not a flood predictor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_06_proxy_baseline_schema_v1.json"
DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_v1.csv"
DS_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_06_proxy_baseline_dataset_manifest_v1.json"
)
OUT_DIR = ROOT / "outputs_public" / "suscetibilidade"
TARGET_POLICY = OUT_DIR / "SUSC_06_proxy_baseline_target_policy.json"
FEATURE_TABLE = OUT_DIR / "SUSC_06_proxy_baseline_feature_table.csv"
METRICS = OUT_DIR / "SUSC_06_proxy_baseline_metrics.csv"
REGION_METRICS = OUT_DIR / "SUSC_06_proxy_baseline_region_metrics.csv"
PARTIAL_EFFECTS = OUT_DIR / "SUSC_06_proxy_baseline_partial_effects.csv"
REPORT = OUT_DIR / "SUSC_06_proxy_gam_spgam_baseline_report.md"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}


@pytest.fixture(scope="module")
def manifest():
    with DS_MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def policy():
    with TARGET_POLICY.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_exists():
    assert SCHEMA.exists()


def test_baseline_dataset_exists():
    assert DATASET.exists()


def test_manifest_exists(manifest):
    assert DS_MANIFEST.exists()
    assert manifest["rows"] == 300


def test_target_policy_exists(policy):
    assert TARGET_POLICY.exists()
    assert policy["baseline_status"] == "proxy_baseline_review_only"


def test_report_exists():
    assert REPORT.exists()


def test_outputs_exist():
    for p in (FEATURE_TABLE, METRICS, REGION_METRICS, PARTIAL_EFFECTS):
        assert p.exists(), f"missing output: {p}"


def test_review_only_invariants(manifest):
    assert manifest["allowed_for_training"] is False
    assert manifest["can_be_used_as_ground_truth"] is False
    assert manifest["review_only"] is True


def test_target_is_proxy(policy):
    assert policy["target_is_proxy"] is True
    assert policy["target_is_ground_truth"] is False


def test_no_ground_truth(manifest):
    assert manifest["ground_truth_targets"] == []


def test_no_persisted_model():
    found = []
    for base in (DATASET.parent, ROOT / "scripts" / "suscetibilidade", OUT_DIR):
        if base.exists():
            found += [p.name for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    assert not found, f"unexpected model artifact(s): {found}"


def test_mandatory_statement_in_report():
    text = REPORT.read_text(encoding="utf-8")
    assert "não valida ocorrência real de enchente" in text
    assert "não cria ground truth" in text
