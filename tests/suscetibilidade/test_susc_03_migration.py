"""
SUSC-03 tests — migration, governance, and exploratory profiling.

These tests assert that the SUSC-03 artifacts exist, are governed (no
training, no ground truth), and are internally consistent (SHA256). They do
NOT train any model and do NOT treat heuristic labels or v5 scores as truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_PATH = ROOT / "schemas" / "suscetibilidade" / "susc_features_schema_v1.json"
CSV_PATH = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
ARTIFACT_PATH = (
    ROOT
    / "manifests"
    / "suscetibilidade"
    / "susc_features_by_patch_v1_artifact_manifest.json"
)
REPORT_PATH = (
    ROOT
    / "outputs_public"
    / "suscetibilidade"
    / "SUSC_03_dataset_migration_and_profile_report.md"
)
ROADMAP_PATH = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_roadmap_after_03.md"
PROFILE_OUTPUTS = [
    "SUSC_03_feature_profile_by_group.csv",
    "SUSC_03_missingness_report.csv",
    "SUSC_03_numeric_summary.csv",
    "SUSC_03_region_summary.csv",
    "SUSC_03_feature_correlation_screen.csv",
]
EXPECTED_ROWS = 300


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def artifact() -> dict:
    with ARTIFACT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def csv_rows() -> tuple[list[str], list[list[str]]]:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, list(reader)


def test_core_artifacts_exist():
    for p in (SCHEMA_PATH, CSV_PATH, ARTIFACT_PATH, REPORT_PATH, ROADMAP_PATH):
        assert p.exists(), f"missing artifact: {p}"


def test_profile_outputs_exist():
    out_dir = ROOT / "outputs_public" / "suscetibilidade"
    for name in PROFILE_OUTPUTS:
        assert (out_dir / name).exists(), f"missing profile output: {name}"


def test_csv_has_300_rows(csv_rows):
    _, data = csv_rows
    assert len(data) == EXPECTED_ROWS


def test_csv_has_patch_identifier(csv_rows):
    header, _ = csv_rows
    assert "patch_id" in header


def test_artifact_manifest_sha256_matches(artifact):
    assert artifact["sha256"] == _sha256(CSV_PATH)


def test_artifact_manifest_row_col_counts(artifact, csv_rows):
    header, data = csv_rows
    assert artifact["rows"] == len(data) == EXPECTED_ROWS
    assert artifact["columns"] == len(header)


def test_no_feature_allows_training(schema, csv_rows):
    header, _ = csv_rows
    by_col = {f["source_column"]: f for f in schema["features"]}
    for col in header:
        feat = by_col.get(col)
        if feat is not None:
            assert feat.get("allowed_for_training") is False, col


def test_no_feature_is_ground_truth(schema, csv_rows):
    header, _ = csv_rows
    by_col = {f["source_column"]: f for f in schema["features"]}
    for col in header:
        feat = by_col.get(col)
        if feat is not None:
            assert feat.get("can_be_used_as_ground_truth") is False, col


def test_artifact_governance_flags(artifact):
    assert artifact["allowed_for_training"] is False
    assert artifact["review_only"] is True
    assert artifact["can_be_used_as_ground_truth"] is False
    assert artifact["scientific_status"] == "susceptibility_features_not_event_ground_truth"


def test_report_contains_mandatory_disclaimer():
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "não constitui ground truth de ocorrência" in text
    assert "não desbloqueia treinamento supervisionado" in text


def test_heuristic_labels_and_scores_not_gt(schema, csv_rows):
    header, _ = csv_rows
    by_col = {f["source_column"]: f for f in schema["features"]}
    for col in header:
        feat = by_col.get(col)
        if feat and feat["group"] in {"heuristic_label", "score"}:
            assert feat.get("can_be_used_as_ground_truth") is False, col
