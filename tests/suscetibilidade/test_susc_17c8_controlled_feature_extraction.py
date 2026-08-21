"""SUSC-17C8 controlled candidate feature extraction tests."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED_OUTPUTS = [
    OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C8.md",
    OUT / "SUSC_17C8_EXTRACAO_REAL_CONTROLADA_FEATURES_PATCHES_CANDIDATOS_REPORT.md",
    OUT / "susc_17c8_extraction_run_registry.csv",
    OUT / "susc_17c8_physical_static_features.csv",
    OUT / "susc_17c8_urban_territorial_features.csv",
    OUT / "susc_17c8_rainfall_trigger_features.csv",
    OUT / "susc_17c8_candidate_feature_matrix_partial.csv",
    OUT / "susc_17c8_feature_quality_flags.csv",
    OUT / "susc_17c8_feature_provenance_manifest.csv",
    OUT / "susc_17c8_no_leakage_audit.csv",
    OUT / "susc_17c8_readiness_summary.json",
    OUT / "susc_17c8_promotion_blockers.csv",
    SCHEMAS / "susc_17c8_extracted_feature_schema_v1.json",
    SCHEMAS / "susc_17c8_feature_provenance_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c8_controlled_feature_extraction_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c8_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_build_generates_and_validates_all_artifacts():
    result = run_script("build_susc_17c8_controlled_feature_extraction.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in EXPECTED_OUTPUTS:
        assert path.exists(), path
    result = run_script("validate_susc_17c8_controlled_feature_extraction.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_valid_for_feature_and_provenance_rows():
    s17c8 = _load_common()
    feature_schema = json.loads((SCHEMAS / "susc_17c8_extracted_feature_schema_v1.json").read_text(encoding="utf-8"))
    provenance_schema = json.loads((SCHEMAS / "susc_17c8_feature_provenance_schema_v1.json").read_text(encoding="utf-8"))
    for path in (
        OUT / "susc_17c8_physical_static_features.csv",
        OUT / "susc_17c8_urban_territorial_features.csv",
        OUT / "susc_17c8_rainfall_trigger_features.csv",
    ):
        for row in read_csv(path):
            assert s17c8._schema_violations(row, feature_schema) == []
    for row in read_csv(OUT / "susc_17c8_feature_provenance_manifest.csv"):
        assert s17c8._schema_violations(row, provenance_schema) == []


def test_build_twice_byte_identical():
    first = tuple(path.read_bytes() for path in EXPECTED_OUTPUTS)
    result = run_script("build_susc_17c8_controlled_feature_extraction.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in EXPECTED_OUTPUTS)
    assert first == second


def test_no_score_v7_score_v6_or_official_dataset_change():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    summary = json.loads((OUT / "susc_17c8_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["official_patch_created"] is False
    assert summary["official_patch_link_created"] is False
    for path in (
        "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv",
        "datasets/suscetibilidade/susc_features_by_patch_v1.csv",
    ):
        changed = subprocess.run(
            ["git", "diff", "--name-only", "--", path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert changed.stdout.strip() == ""


def test_no_synthetic_as_real_training_label_model_or_ground_truth():
    for path in (
        OUT / "susc_17c8_physical_static_features.csv",
        OUT / "susc_17c8_urban_territorial_features.csv",
        OUT / "susc_17c8_rainfall_trigger_features.csv",
        OUT / "susc_17c8_candidate_feature_matrix_partial.csv",
        OUT / "susc_17c8_extraction_run_registry.csv",
        OUT / "susc_17c8_promotion_blockers.csv",
    ):
        for row in read_csv(path):
            assert row["review_only"] == "true"
            assert row["trainable"] == "false"
            assert row["ground_truth"] == "false"
    for path in (
        OUT / "susc_17c8_physical_static_features.csv",
        OUT / "susc_17c8_urban_territorial_features.csv",
        OUT / "susc_17c8_rainfall_trigger_features.csv",
    ):
        for row in read_csv(path):
            assert row["feature_value_real"] == "false"
            assert row["synthetic_placeholder"] == "false"
            assert row["usable_for_science_now"] == "false"


def test_counts_layers_and_no_raw_raster_commit():
    physical = read_csv(OUT / "susc_17c8_physical_static_features.csv")
    urban = read_csv(OUT / "susc_17c8_urban_territorial_features.csv")
    rainfall = read_csv(OUT / "susc_17c8_rainfall_trigger_features.csv")
    assert len(physical) == 40
    assert len(urban) == 30
    assert len(rainfall) == 20
    assert {row["raw_artifact_committed"] for row in read_csv(OUT / "susc_17c8_feature_provenance_manifest.csv")} == {"false"}
    assert {row["raw_artifact_committed"] for row in read_csv(OUT / "susc_17c8_extraction_run_registry.csv")} == {"false"}


def test_sentinel_sar_embedding_not_extracted_and_17b_blocked():
    summary = json.loads((OUT / "susc_17c8_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["sentinel2_spectral_extracted"] is False
    assert summary["sar_observational_extracted"] is False
    assert summary["embedding_representation_extracted"] is False
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    for row in read_csv(OUT / "susc_17c8_candidate_feature_matrix_partial.csv"):
        assert row["has_sentinel2_spectral_real"] == "false"
        assert row["has_sar_observational_real"] == "false"
        assert row["has_embedding_representation_real"] == "false"
        assert row["real_feature_layer_count"] == "0"


def test_no_leakage_policy_passes_without_post_event_misuse():
    for row in read_csv(OUT / "susc_17c8_no_leakage_audit.csv"):
        assert row["uses_post_event_data"] == "false"
        assert row["uses_reference_footprint_as_feature"] == "false"
        assert row["uses_sar_post_event_as_feature"] == "false"
        assert row["uses_synthetic_value_as_real"] == "false"
        assert row["passes_no_leakage_policy"] == "true"
