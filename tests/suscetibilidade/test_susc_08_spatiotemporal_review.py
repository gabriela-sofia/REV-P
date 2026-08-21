"""
SUSC-08 tests — spatiotemporal human review package.

Assert the casebook/checklist/form/report exist, the SUSC-07B spatial cases are
present, governance is preserved (no GT, no training), every spatial case
requires human review, the form initializes ground truth false, the SUSC-03
matrix is unchanged, and no model is persisted.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())

SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_08_spatiotemporal_review_schema_v1.json"
CASES = ROOT / "datasets" / "suscetibilidade" / "susc_08_spatiotemporal_review_cases_v1.csv"
CASEBOOK = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_review_casebook.md"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_review_casebook_summary.csv"
CHECKLIST = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_manual_review_checklist.md"
FORM = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_manual_review_form.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_08_spatiotemporal_review_report.md"
MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ART_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_features_by_patch_v1_artifact_manifest.json"

MODEL_EXTS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx", ".ckpt", ".sav", ".model"}
SPATIAL = {"bbox_overlap", "near_patch_buffer_candidate", "exact_patch_overlap"}


def load_csv(p):
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def cases():
    return load_csv(CASES)


@pytest.fixture(scope="module")
def form():
    return load_csv(FORM)


def test_artifacts_exist():
    for p in (SCHEMA, CASES, CASEBOOK, SUMMARY, CHECKLIST, FORM, REPORT):
        assert p.exists(), f"missing: {p}"


def test_spatial_cases_present(cases):
    spatial = [c for c in cases if c["spatial_relation"] in SPATIAL]
    assert len(spatial) >= 9, f"expected >=9 spatial cases, got {len(spatial)}"
    patches = {c["patch_id"] for c in spatial}
    assert "recife_00276" in patches
    assert "petropolis_00467" in patches


def test_governance_preserved(cases):
    for c in cases:
        assert c["can_be_ground_truth"] == "false", c["case_id"]
        assert c["allowed_for_training"] == "false", c["case_id"]
        assert c["review_only"] == "true", c["case_id"]


def test_no_ground_truth(cases):
    assert all(c["can_be_ground_truth"] == "false" for c in cases)


def test_no_training(cases):
    assert all(c["allowed_for_training"] == "false" for c in cases)


def test_every_spatial_case_requires_human_review(cases):
    for c in cases:
        if c["spatial_relation"] in SPATIAL:
            assert c["requires_human_review"] == "true", c["case_id"]


def test_no_strong_candidate_without_footprint(cases):
    assert all(c["case_strength_pre_review"] != "strong_candidate" for c in cases)


def test_form_initializes_ground_truth_false(form):
    assert form
    for r in form:
        assert r["approved_for_ground_truth"] == "false", r["case_id"]


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
