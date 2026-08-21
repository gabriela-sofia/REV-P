"""SUSC-17C6 multimodal applicability canary tests.

This package must remain review-only: no official patch, no official patch-link,
no score v7, no score v6 change, no training, no label and no ground truth.
"""

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

REPORT = OUT / "SUSC_17C6_MULTIMODAL_APPLICABILITY_CANARY_REPORT.md"
CANDIDATE_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
CANDIDATE_GRID_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
CANDIDATE_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"
FEATURE_CONTRACT = OUT / "susc_17c6_multimodal_feature_contract.csv"
CANARY_MATRIX = OUT / "susc_17c6_multimodal_canary_matrix.csv"
EMBEDDING_CONTRACT = OUT / "susc_17c6_embedding_contract.json"
SYNTHETIC_EMBEDDING = OUT / "susc_17c6_synthetic_embedding_smoke_test.csv"
SUMMARY = OUT / "susc_17c6_applicability_readiness_summary.json"
BLOCKERS = OUT / "susc_17c6_promotion_blockers.csv"
MATRIX_SCHEMA = SCHEMAS / "susc_17c6_multimodal_canary_schema_v1.json"
EMBEDDING_SCHEMA = SCHEMAS / "susc_17c6_embedding_contract_schema_v1.json"


def _load_common():
    path = SCRIPTS / "susc_17c6_multimodal_canary_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c6_common", path)
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


def test_build_generates_all_artifacts():
    result = run_script("build_susc_17c6_multimodal_applicability_canary.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in (
        REPORT,
        CANDIDATE_GRID,
        CANDIDATE_GRID_GEOJSON,
        CANDIDATE_LINKS,
        FEATURE_CONTRACT,
        CANARY_MATRIX,
        EMBEDDING_CONTRACT,
        SYNTHETIC_EMBEDDING,
        SUMMARY,
        BLOCKERS,
    ):
        assert path.exists(), path
    result = run_script("validate_susc_17c6_multimodal_applicability_canary.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_valid():
    s = _load_common()
    matrix_schema = json.loads(MATRIX_SCHEMA.read_text(encoding="utf-8"))
    embedding_schema = json.loads(EMBEDDING_SCHEMA.read_text(encoding="utf-8"))
    for row in read_csv(CANARY_MATRIX):
        assert s._schema_violations(row, matrix_schema) == []
    contract = json.loads(EMBEDDING_CONTRACT.read_text(encoding="utf-8"))
    for field in embedding_schema["required"]:
        assert field in contract


def test_ids_deterministic_and_build_twice_byte_identical():
    outputs = (
        REPORT,
        CANDIDATE_GRID,
        CANDIDATE_GRID_GEOJSON,
        CANDIDATE_LINKS,
        FEATURE_CONTRACT,
        CANARY_MATRIX,
        EMBEDDING_CONTRACT,
        SYNTHETIC_EMBEDDING,
        SUMMARY,
        BLOCKERS,
    )
    first = tuple(p.read_bytes() for p in outputs)
    result = run_script("build_susc_17c6_multimodal_applicability_canary.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(p.read_bytes() for p in outputs)
    assert first == second
    for path, key in (
        (CANDIDATE_GRID, "candidate_patch_id"),
        (CANDIDATE_LINKS, "candidate_patch_link_id"),
        (CANARY_MATRIX, "canary_row_id"),
        (SYNTHETIC_EMBEDDING, "synthetic_embedding_id"),
        (BLOCKERS, "blocker_id"),
    ):
        ids = [row[key] for row in read_csv(path)]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)


def test_no_score_v7_and_score_v6_unchanged():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["score_v7_created"] is False
    assert summary["score_v6_changed"] is False
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert changed.stdout.strip() == ""


def test_no_trainable_ground_truth_or_scientific_synthetic_values():
    for path in (CANDIDATE_GRID, CANDIDATE_LINKS, CANARY_MATRIX, BLOCKERS):
        for row in read_csv(path):
            assert row["review_only"] == "true"
            assert row["trainable"] == "false"
            assert row["ground_truth"] == "false"
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["review_only"] is True
    assert summary["trainable"] is False
    assert summary["ground_truth"] is False
    assert summary["synthetic_values_marked_scientific"] is False


def test_no_official_patch_or_patch_link_created():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["official_patch_created"] is False
    assert summary["official_patch_link_created"] is False
    assert subprocess.run(
        ["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_features_by_patch_v1.csv"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() == ""
    for row in read_csv(CANDIDATE_GRID):
        assert row["official_patch"] == "false"
        assert row["candidate_patch_id"].startswith("S17C6_CANARY_REC_")
        assert row["candidate_patch_id"] != "REC_00019"
    for row in read_csv(CANDIDATE_LINKS):
        assert row["official_patch_link"] == "false"


def test_multimodal_contract_contains_required_layers_and_missingness():
    layers = {row["layer"] for row in read_csv(FEATURE_CONTRACT)}
    assert layers == {
        "physical_static",
        "urban_territorial",
        "sentinel2_spectral",
        "rainfall_trigger",
        "documentary_evidence",
        "sar_observational",
        "embedding_representation",
    }
    for row in read_csv(CANARY_MATRIX):
        missing = set(row["missing_layers"].split(";"))
        assert "physical_static" in missing
        assert "urban_territorial" in missing
        assert "sentinel2_spectral" in missing
        assert "rainfall_trigger" in missing
        assert "sar_observational" in missing
        assert "embedding_representation" in missing


def test_embedding_is_synthetic_only():
    contract = json.loads(EMBEDDING_CONTRACT.read_text(encoding="utf-8"))
    assert contract["real_embedding_created"] is False
    assert contract["synthetic_embedding_smoke_test_created"] is True
    assert contract["not_scientific_until_real_tile"] is True
    for row in read_csv(SYNTHETIC_EMBEDDING):
        assert row["synthetic_embedding"] == "true"
        assert row["usable_for_science"] == "false"
        assert row["usable_for_training"] == "false"
        assert row["review_only"] == "true"
        assert row["ground_truth"] == "false"
        assert len(row["embedding_hash"]) == 64


def test_17b_and_score_v7_ineligible_with_nonempty_blockers():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["candidate_patch_count"] == len(read_csv(CANDIDATE_GRID))
    assert summary["candidate_patch_link_count"] == len(read_csv(CANDIDATE_LINKS))
    assert summary["milestone_category"] == "canario_de_aplicabilidade_multimodal_review_only"
    blockers = read_csv(BLOCKERS)
    assert blockers
    assert {row["blocker_code"] for row in blockers} >= {
        "candidate_patches_not_official",
        "candidate_patch_features_not_extracted",
        "no_real_embedding_tile",
        "synthetic_embedding_not_scientific",
        "no_sar_runtime",
        "p0_official_packages_missing",
    }
