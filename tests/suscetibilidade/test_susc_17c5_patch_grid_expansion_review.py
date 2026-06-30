"""SUSC-17C5 Patch Grid Expansion Review tests.

The package must stay review-only and fail-closed: no score v7, no score v6
change, no official patch, no official patch-link, no namespace promotion for
REC_00019, AOI is not a patch or footprint, and 17B remains blocked.
"""

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

GRID = OUT / "susc_17c5_patch_grid_inventory.csv"
COVERAGE = OUT / "susc_17c5_charter758_grid_coverage_audit.csv"
NAMESPACE = OUT / "susc_17c5_protocolc_to_susc_patch_namespace_audit.csv"
AOI = OUT / "susc_17c5_candidate_expansion_aoi.geojson"
OPTIONS = OUT / "susc_17c5_patch_grid_expansion_options.csv"
POLICY = OUT / "susc_17c5_expansion_risk_policy.json"
NEXT_ACTIONS = OUT / "susc_17c5_next_action_decision_table.csv"
SUMMARY = OUT / "susc_17c5_summary.json"
REPORT = OUT / "SUSC_17C5_PATCH_GRID_EXPANSION_REVIEW_REPORT.md"
GRID_SCHEMA = SCHEMAS / "susc_17c5_patch_grid_inventory_schema_v1.json"
OPTION_SCHEMA = SCHEMAS / "susc_17c5_expansion_option_schema_v1.json"


def _load_common():
    path = SCRIPTS / "susc_17c5_patch_grid_expansion_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c5_common", path)
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
    result = run_script("build_susc_17c5_patch_grid_expansion_review.py")
    assert result.returncode == 0, result.stderr + result.stdout
    for path in (GRID, COVERAGE, NAMESPACE, AOI, OPTIONS, POLICY, NEXT_ACTIONS, SUMMARY, REPORT):
        assert path.exists(), path
    result = run_script("validate_susc_17c5_patch_grid_expansion_review.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_schemas_valid():
    s = _load_common()
    grid_schema = json.loads(GRID_SCHEMA.read_text(encoding="utf-8"))
    option_schema = json.loads(OPTION_SCHEMA.read_text(encoding="utf-8"))
    for row in read_csv(GRID):
        assert s._schema_violations(row, grid_schema) == []
    for row in read_csv(OPTIONS):
        assert s._schema_violations(row, option_schema) == []


def test_ids_deterministic():
    for path, key in (
        (GRID, "patch_grid_id"),
        (COVERAGE, "coverage_audit_id"),
        (NAMESPACE, "namespace_audit_id"),
        (OPTIONS, "expansion_option_id"),
        (NEXT_ACTIONS, "decision_id"),
    ):
        ids = [r[key] for r in read_csv(path)]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)


def test_build_twice_byte_identical():
    run_script("build_susc_17c5_patch_grid_expansion_review.py")
    outputs = (GRID, COVERAGE, NAMESPACE, AOI, OPTIONS, POLICY, NEXT_ACTIONS, SUMMARY, REPORT)
    first = tuple(p.read_bytes() for p in outputs)
    run_script("build_susc_17c5_patch_grid_expansion_review.py")
    second = tuple(p.read_bytes() for p in outputs)
    assert first == second


def test_no_score_v7_and_score_v6_unchanged():
    assert not (ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv").exists()
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["score_v7_created"] is False
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["score_v6_changed"] is False
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_score_v6_candidate_by_patch_v1.csv"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert changed.stdout.strip() == ""


def test_no_trainable_or_ground_truth():
    for path in (GRID, COVERAGE, OPTIONS, NEXT_ACTIONS):
        for row in read_csv(path):
            assert row["review_only"] == "true"
            assert row["trainable"] == "false"
            assert row["ground_truth"] == "false"
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["review_only"] is True
    assert s["trainable"] is False
    assert s["ground_truth"] is False


def test_no_official_patch_or_patch_link_created():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["official_patch_created"] is False
    assert s["official_patch_link_created"] is False
    assert subprocess.run(
        ["git", "diff", "--name-only", "--", "datasets/suscetibilidade/susc_features_by_patch_v1.csv"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() == ""
    assert read_csv(COVERAGE)[0]["intersecting_patch_count"] == "0"


def test_rec_00019_does_not_become_susc_patch():
    row = read_csv(NAMESPACE)[0]
    assert row["source_patch_id"] == "REC_00019"
    assert row["target_susc_patch_match"] == "false"
    assert row["match_status"] == "namespace_incompatible"


def test_aoi_is_not_patch_or_footprint():
    geojson = json.loads(AOI.read_text(encoding="utf-8"))
    roles = [f["properties"]["feature_role"] for f in geojson["features"]]
    assert roles == [
        "observed_reference_geometry",
        "current_susc_grid_extent",
        "candidate_expansion_envelope",
        "candidate_review_buffer",
    ]
    for feature in geojson["features"]:
        props = feature["properties"]
        assert props["review_only"] is True
        assert props["ground_truth"] is False
        assert props["not_official_patch"] is True
        assert props["not_footprint"] is True
        assert props["not_patch_link"] is True


def test_options_that_require_new_patch_or_score_are_blocked():
    for row in read_csv(OPTIONS):
        if row["requires_new_patches"] == "true" or row["requires_score_recompute"] == "true":
            assert row["allowed_now"] == "false"
    bridge = [r for r in read_csv(OPTIONS) if r["option_name"] == "create_protocolc_to_susc_bridge_only"][0]
    assert bridge["allowed_now"] == "false"


def test_17b_ineligible_without_patch_link_and_qa():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert s["eligible_for_17b_now"] is False
    assert "no_valid_susc_patch_link" in s["blocks_17b"]
    for row in read_csv(NEXT_ACTIONS):
        assert row["can_unlock_17b"] == "false"


def test_summary_reflects_real_distance_and_intersection():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cov = read_csv(COVERAGE)[0]
    assert s["patch_grids_inventoried"] == 3
    assert s["recife_patch_count"] == 100
    assert s["charter758_intersecting_patch_count"] == 0
    assert round(s["charter758_nearest_patch_distance_m"], 1) == round(float(cov["nearest_patch_distance_m"]), 1)
    assert s["charter758_coverage_status"] == cov["coverage_status"]


def test_policy_blocks_silent_expansion():
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    blocked = policy["blocked_actions"]
    assert blocked["silent_grid_expansion"] is True
    assert blocked["patch_creation_without_manifest"] is True
    assert blocked["patch_link_without_official_patch"] is True
    assert blocked["protocolc_susc_namespace_mixing"] is True
    assert blocked["benchmark_17b_without_qa_accepted_and_valid_patch_link"] is True
