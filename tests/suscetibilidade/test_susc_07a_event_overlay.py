"""
SUSC-07A tests — event evidence catalog + patch overlay, review-only, no GT.

Assert the inventory/catalog/overlay artifacts exist, carry the required columns,
preserve governance (no ground truth, no training), keep the SUSC-03 matrix
intact, and persist no model. Documentary evidence != patch ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_07_event_evidence_schema_v1.json"
SCAN = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_event_source_scan.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_07_event_evidence_catalog_v1.csv"
CAT_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_07_event_evidence_catalog_manifest_v1.json"
)
OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_candidates.csv"
OVERLAY_SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_summary.csv"
OVERLAY_LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_patch_event_overlay_limitations.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07A_event_evidence_overlay_report.md"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ARTIFACT_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_features_by_patch_v1_artifact_manifest.json"
)

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}

CATALOG_COLS = {
    "evidence_id", "region", "event_type", "date_or_period", "source_path",
    "has_geometry", "geometry_status", "spatial_precision", "confidence_level",
    "can_link_to_patch", "can_be_ground_truth", "allowed_for_training",
    "review_only", "evidence_status",
}
OVERLAY_COLS = {
    "patch_id", "region", "susceptibility_proxy", "evidence_id", "evidence_status",
    "spatial_relation", "temporal_relation", "overlay_confidence",
    "can_be_used_as_ground_truth", "allowed_for_training", "review_only",
    "interpretation", "limitation",
}


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def catalog():
    return load_csv(CATALOG)


@pytest.fixture(scope="module")
def overlay():
    return load_csv(OVERLAY)


def test_artifacts_exist():
    for p in (SCHEMA, SCAN, CATALOG, CAT_MANIFEST, OVERLAY, OVERLAY_SUMMARY,
              OVERLAY_LIMITS, REPORT):
        assert p.exists(), f"missing: {p}"


def test_catalog_has_required_columns(catalog):
    assert catalog
    assert CATALOG_COLS.issubset(set(catalog[0].keys()))


def test_overlay_has_required_columns(overlay):
    assert overlay
    assert OVERLAY_COLS.issubset(set(overlay[0].keys()))


def test_governance_preserved(catalog, overlay):
    for r in catalog:
        assert r["can_be_ground_truth"] == "false", r["evidence_id"]
        assert r["allowed_for_training"] == "false", r["evidence_id"]
        assert r["review_only"] == "true", r["evidence_id"]
    for r in overlay:
        assert r["can_be_used_as_ground_truth"] == "false"
        assert r["allowed_for_training"] == "false"
        assert r["review_only"] == "true"


def test_ground_truth_stays_false(catalog):
    assert all(r["can_be_ground_truth"] == "false" for r in catalog)


def test_training_stays_false(catalog):
    assert all(r["allowed_for_training"] == "false" for r in catalog)


def test_report_has_mandatory_statement():
    text = REPORT.read_text(encoding="utf-8")
    assert "não cria ground truth de enchente por patch" in text


def test_matrix_exists_and_unmodified():
    assert MATRIX.exists()
    with ARTIFACT_MANIFEST.open("r", encoding="utf-8") as f:
        expected = json.load(f)["sha256"]
    h = hashlib.sha256()
    with MATRIX.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    assert h.hexdigest() == expected, "SUSC-03 matrix was modified"


def test_no_persisted_model():
    found = []
    for base in (ROOT / "outputs_public" / "suscetibilidade",
                 ROOT / "scripts" / "suscetibilidade",
                 ROOT / "datasets" / "suscetibilidade"):
        if base.exists():
            found += [p.name for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    assert not found, f"unexpected model artifact(s): {found}"


def test_no_evidence_linkable_without_geometry(catalog):
    for r in catalog:
        if r["can_link_to_patch"] == "true":
            assert r["has_geometry"] == "true" or r["spatial_precision"] in {"patch", "neighborhood"}
