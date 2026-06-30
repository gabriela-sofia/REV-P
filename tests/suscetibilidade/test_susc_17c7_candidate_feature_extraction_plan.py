"""SUSC-17C7 candidate feature extraction plan tests."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

REPORT = OUT / "SUSC_17C7_PLANO_EXTRACAO_FEATURES_PATCHES_CANDIDATOS_REPORT.md"
FEATURE_INVENTORY = OUT / "susc_17c7_candidate_patch_feature_inventory.csv"
SOURCE_MAPPING = OUT / "susc_17c7_feature_source_mapping.csv"
TASK_PLAN = OUT / "susc_17c7_extraction_task_plan.csv"
MISSINGNESS = OUT / "susc_17c7_feature_missingness_matrix.csv"
EMBEDDING_READINESS = OUT / "susc_17c7_embedding_input_readiness.csv"
NO_LEAKAGE_POLICY = OUT / "susc_17c7_no_leakage_policy.json"
SUMMARY = OUT / "susc_17c7_readiness_summary.json"
BLOCKERS = OUT / "susc_17c7_promotion_blockers.csv"
INVENTORY_SCHEMA = SCHEMAS / "susc_17c7_feature_inventory_schema_v1.json"
TASK_SCHEMA = SCHEMAS / "susc_17c7_extraction_task_schema_v1.json"


def _load_common():
    path = SCRIPTS / "susc_17c7_candidate_feature_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c7_common", path)
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
    result = run_script("build_susc_17c7_candidate_feature_extraction_plan.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in (
        REPORT,
        FEATURE_INVENTORY,
        SOURCE_MAPPING,
        TASK_PLAN,
        MISSINGNESS,
        EMBEDDING_READINESS,
        NO_LEAKAGE_POLICY,
        SUMMARY,
        BLOCKERS,
    ):
        assert path.exists(), path
    result = run_script("validate_susc_17c7_candidate_feature_extraction_plan.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_valid():
    s = _load_common()
    inventory_schema = json.loads(INVENTORY_SCHEMA.read_text(encoding="utf-8"))
    task_schema = json.loads(TASK_SCHEMA.read_text(encoding="utf-8"))
    for row in read_csv(FEATURE_INVENTORY):
        assert s._schema_violations(row, inventory_schema) == []
    for row in read_csv(TASK_PLAN):
        assert s._schema_violations(row, task_schema) == []


def test_build_twice_byte_identical_and_ids_deterministic():
    outputs = (
        REPORT,
        FEATURE_INVENTORY,
        SOURCE_MAPPING,
        TASK_PLAN,
        MISSINGNESS,
        EMBEDDING_READINESS,
        NO_LEAKAGE_POLICY,
        SUMMARY,
        BLOCKERS,
    )
    first = tuple(path.read_bytes() for path in outputs)
    result = run_script("build_susc_17c7_candidate_feature_extraction_plan.py")
    assert result.returncode == 0, result.stderr + result.stdout
    second = tuple(path.read_bytes() for path in outputs)
    assert first == second
    task_ids = [row["extraction_task_id"] for row in read_csv(TASK_PLAN)]
    assert len(task_ids) == len(set(task_ids))
    assert task_ids == sorted(task_ids)


def test_no_score_v7_score_v6_change_or_official_patch_change():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
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


def test_no_trainable_ground_truth_or_real_synthetic_feature():
    for path in (FEATURE_INVENTORY, TASK_PLAN, BLOCKERS):
        for row in read_csv(path):
            assert row["review_only"] == "true"
            assert row["trainable"] == "false"
            assert row["ground_truth"] == "false"
    for row in read_csv(FEATURE_INVENTORY):
        assert row["usable_for_science_now"] == "false"
        if row["synthetic_placeholder"] == "true":
            assert row["available_now"] == "false"


def test_all_candidate_patches_and_layers_are_present():
    candidate_ids = {row["candidate_patch_id"] for row in read_csv(OUT / "susc_17c6_candidate_patch_grid.csv")}
    layers = {
        "physical_static",
        "urban_territorial",
        "sentinel2_spectral",
        "rainfall_trigger",
        "documentary_evidence",
        "sar_observational",
        "embedding_representation",
    }
    assert {row["candidate_patch_id"] for row in read_csv(MISSINGNESS)} == candidate_ids
    assert {row["feature_layer"] for row in read_csv(SOURCE_MAPPING)} == layers
    inventory_pairs = {(row["candidate_patch_id"], row["feature_layer"]) for row in read_csv(FEATURE_INVENTORY)}
    for patch_id in candidate_ids:
        for layer in layers:
            assert (patch_id, layer) in inventory_pairs
    for row in read_csv(MISSINGNESS):
        missing = set(row["missing_layers"].split(";"))
        assert layers - {"documentary_evidence"} <= missing
        assert row["has_documentary_evidence"] == "true"


def test_embedding_real_not_executed_and_synthetic_not_scientific():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["real_embedding_input_ready_count"] == 0
    assert summary["synthetic_embedding_only_count"] == len(read_csv(EMBEDDING_READINESS))
    for row in read_csv(EMBEDDING_READINESS):
        assert row["has_real_tile"] == "false"
        assert row["embedding_input_ready"] == "false"
        assert row["embedding_can_run_now"] == "false"
        assert row["synthetic_embedding_exists_from_17c6"] == "true"
        assert row["synthetic_embedding_usable_for_science"] == "false"


def test_no_leakage_policy_blocks_required_uses():
    policy = json.loads(NO_LEAKAGE_POLICY.read_text(encoding="utf-8"))
    blocked = policy["blocked_uses"]
    assert blocked["post_event_footprint_as_feature"] is True
    assert blocked["post_event_sar_as_pre_event_feature"] is True
    assert blocked["documentary_evidence_as_physical_susceptibility_proxy"] is True
    assert blocked["synthetic_placeholder_as_real_feature"] is True
    assert blocked["candidate_patch_as_official_patch"] is True
    assert blocked["REC_00019_as_SUSC_patch"] is True
    assert blocked["promotion_to_17b_without_qa_accepted"] is True
    assert blocked["score_v7_without_readiness"] is True
    assert blocked["training_without_future_ground_truth_policy"] is True


def test_17b_score_v7_blocked_and_summary_counts_real():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    inventory = read_csv(FEATURE_INVENTORY)
    assert summary["candidate_patch_count"] == len(read_csv(OUT / "susc_17c6_candidate_patch_grid.csv"))
    assert summary["feature_rows_count"] == len(inventory)
    assert summary["eligible_for_17b_now"] is False
    assert summary["eligible_for_score_v7"] is False
    assert summary["eligible_for_training"] is False
    assert read_csv(BLOCKERS)
