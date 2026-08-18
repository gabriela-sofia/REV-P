"""
SUSC-07B tests — real coordinate extraction, enrichment, refined overlay.

Assert artifacts exist with required columns, governance is preserved (no GT, no
training), no forbidden geometry source is used, linkable coordinates carry
explicit geometry, the SUSC-03 matrix is unchanged, and no model is persisted.
Real traceable coordinates only; no invention; not patch ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_07b_event_geometry_acquisition_schema_v1.json"
SCAN = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_real_coordinate_source_scan.csv"
QUEUE = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_external_coordinate_acquisition_queue.csv"
ACQ_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_07b_external_coordinate_acquisition_manifest_v1.csv"
COORDS = ROOT / "datasets" / "suscetibilidade" / "susc_07b_real_event_coordinates_v1.csv"
ENRICHED = ROOT / "datasets" / "suscetibilidade" / "susc_07b_event_evidence_geometry_enriched_v1.csv"
OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_refined_patch_event_overlay.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_real_coordinate_enrichment_report.md"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ART_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_features_by_patch_v1_artifact_manifest.json"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}
FORBIDDEN = ("centroid", "centroide", "geocod", "manual_guess", "visual_estimate", "unreferenced")

COORD_COLS = {
    "coordinate_id", "evidence_id", "region", "event_type", "date_or_period",
    "source_path_or_url", "source_name", "geometry_type", "lat", "lon", "bbox",
    "wkt", "geojson_ref", "coordinate_crs", "coordinate_precision",
    "extraction_method", "source_confidence", "can_link_to_patch",
    "can_be_ground_truth", "allowed_for_training", "review_only",
    "requires_manual_review",
}
ENRICHED_COLS = {
    "evidence_id", "region", "original_geometry_status", "enriched_has_geometry",
    "enriched_geometry_type", "enriched_geometry_source", "enriched_bbox",
    "enriched_spatial_precision", "can_link_to_patch_after_enrichment",
    "can_be_ground_truth", "allowed_for_training", "review_only",
}
OVERLAY_COLS = {
    "patch_id", "region", "evidence_id", "spatial_relation", "overlay_confidence",
    "can_be_used_as_ground_truth", "allowed_for_training", "review_only",
    "interpretation", "limitation",
}


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def coords():
    return load_csv(COORDS)


@pytest.fixture(scope="module")
def enriched():
    return load_csv(ENRICHED)


@pytest.fixture(scope="module")
def overlay():
    return load_csv(OVERLAY)


def test_artifacts_exist():
    for p in (SCHEMA, SCAN, QUEUE, ACQ_MANIFEST, COORDS, ENRICHED, OVERLAY, REPORT):
        assert p.exists(), f"missing: {p}"


def test_coords_required_columns(coords):
    assert coords
    assert COORD_COLS.issubset(set(coords[0].keys()))


def test_enriched_required_columns(enriched):
    assert enriched
    assert ENRICHED_COLS.issubset(set(enriched[0].keys()))


def test_overlay_required_columns(overlay):
    assert overlay
    assert OVERLAY_COLS.issubset(set(overlay[0].keys()))


def test_ground_truth_false(coords, enriched, overlay):
    for r in coords:
        assert r["can_be_ground_truth"] == "false"
    for r in enriched:
        assert r["can_be_ground_truth"] == "false"
    for r in overlay:
        assert r["can_be_used_as_ground_truth"] == "false"


def test_training_false(coords, enriched, overlay):
    for r in coords + enriched + overlay:
        assert r["allowed_for_training"] == "false"


def test_review_only_preserved(coords, enriched, overlay):
    for r in coords + enriched + overlay:
        assert r["review_only"] == "true"


def test_no_forbidden_source(coords):
    for r in coords:
        blob = (r["source_path_or_url"] + r["extraction_method"] + r["source_confidence"]).lower()
        assert not any(t in blob for t in FORBIDDEN), r["coordinate_id"]


def test_linkable_requires_explicit_geometry(coords):
    for r in coords:
        if r["can_link_to_patch"] == "true":
            assert r["bbox"] or (r["lat"] and r["lon"]) or r["wkt"] or r["geojson_ref"], r["coordinate_id"]


def test_report_has_mandatory_statement():
    text = REPORT.read_text(encoding="utf-8")
    assert "não cria ground truth de enchente por patch" in text


def test_matrix_exists_and_unmodified():
    assert MATRIX.exists()
    expected = json.loads(ART_MANIFEST.read_text(encoding="utf-8"))["sha256"]
    h = hashlib.sha256()
    with MATRIX.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    assert h.hexdigest() == expected


def test_no_persisted_model():
    found = []
    for base in (ROOT / "outputs_public" / "suscetibilidade",
                 ROOT / "scripts" / "suscetibilidade",
                 ROOT / "datasets" / "suscetibilidade"):
        if base.exists():
            found += [p.name for p in base.rglob("*") if p.suffix.lower() in MODEL_EXTS]
    assert not found, f"unexpected model artifact(s): {found}"
