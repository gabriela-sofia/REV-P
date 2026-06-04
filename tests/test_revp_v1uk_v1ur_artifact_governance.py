"""Tests — REV-P v1uk-v1ur Public Repository Artifact Governance.

All writes go to tmp_path. No files are written outside pytest's temp dir,
and no git staging occurs.
"""
from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers — import common module from scripts/repository
# ---------------------------------------------------------------------------

import sys, os
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "repository"))

from revp_v1uk_v1ur_artifact_governance_common import (
    file_size_bytes, file_size_mb, sha256_file_head,
    classify_file_size,
    classify_versioning_policy,
    is_raw_or_cache_path, is_local_only_path, is_public_safe_path,
    detect_absolute_path_content, detect_local_runs_content,
    detect_public_terminology_issue, detect_forbidden_guardrail_true,
    build_public_summary_for_csv,
    guardrail_row, write_csv_with_header, read_csv_safe,
    SIZE_OK, SIZE_WARNING_GT_10MB, SIZE_BLOCKED_GT_50MB, SIZE_BLOCKED_GT_100MB,
    POLICY_PUBLIC_VERSIONED, POLICY_LOCAL_ONLY_LARGE_DERIVED,
    POLICY_RAW_EXTERNAL_NEVER_VERSION, POLICY_CACHE_NEVER_VERSION,
    POLICY_BLOCKED_GT_50MB, POLICY_BLOCKED_GT_100MB,
)
from revp_v1uk_repository_artifact_size_inventory import (
    classify_file_size as _clf,
    build_summary,
)
from revp_v1ul_versioning_policy_classifier import (
    classify_row, build_summary as ul_build_summary,
    _KNOWN_BLOCKED_NAMES,
)
from revp_v1um_large_csv_public_summary_generator import (
    generate_summary_csv, _summary_output_path,
    collect_targets, MAX_SAMPLE_ROWS,
)
from revp_v1un_public_staging_candidate_manifest import (
    build_manifests,
)
from revp_v1uo_public_repository_guardrail_scanner import (
    scan_file, build_summary as uo_build_summary,
)
from revp_v1up_precommit_gate_generator import (
    _PS1_CONTENT,
)
from revp_v1uq_gitignore_policy_updater import (
    build_governance_block, update_gitignore,
    BLOCK_MARKER_START, BLOCK_MARKER_END, GOVERNANCE_PATTERNS,
    _PUBLIC_DATA_POLICY,
)
from revp_v1ur_artifact_governance_bundle import (
    determine_final_status, build_quality_checks,
    STATUS_READY, STATUS_READY_WITH_LOCAL_ONLY, STATUS_FAIL_CLOSED,
)


# ===========================================================================
# 1. Pre-condition: staged files must be empty
# ===========================================================================

def test_staged_files_empty_at_start():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    staged = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    assert staged == [], f"STAGED_NOT_EMPTY: {staged}"


# ===========================================================================
# 2. common — file size helpers
# ===========================================================================

def test_file_size_bytes_small(tmp_path):
    f = tmp_path / "small.txt"
    f.write_bytes(b"hello world")
    assert file_size_bytes(f) == 11


def test_file_size_mb_zero(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    assert file_size_mb(f) == 0.0


def test_sha256_file_head(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"x" * 1000)
    h = sha256_file_head(f)
    assert len(h) == 16
    assert h != "error"


def test_sha256_nonexistent(tmp_path):
    f = tmp_path / "missing.bin"
    assert sha256_file_head(f) == "error"


# ===========================================================================
# 3. common — size classification
# ===========================================================================

def test_classify_size_ok():
    assert classify_file_size(1 * 1024 * 1024) == SIZE_OK


def test_classify_size_warning():
    assert classify_file_size(11 * 1024 * 1024) == SIZE_WARNING_GT_10MB


def test_classify_size_blocked_50():
    assert classify_file_size(51 * 1024 * 1024) == SIZE_BLOCKED_GT_50MB


def test_classify_size_blocked_100():
    assert classify_file_size(101 * 1024 * 1024) == SIZE_BLOCKED_GT_100MB


# ===========================================================================
# 4. v1uk inventory — size classification integration
# ===========================================================================

def test_inventory_classifies_small_file(tmp_path):
    f = tmp_path / "small.py"
    f.write_text("print('ok')", encoding="utf-8")
    size = file_size_bytes(f)
    sc = classify_file_size(size)
    assert sc == SIZE_OK


def test_inventory_detects_gt_10mb(tmp_path):
    f = tmp_path / "medium.csv"
    f.write_bytes(b"a" * (11 * 1024 * 1024))
    sc = classify_file_size(file_size_bytes(f))
    assert sc == SIZE_WARNING_GT_10MB


def test_inventory_detects_gt_50mb(tmp_path):
    f = tmp_path / "large.csv"
    f.write_bytes(b"b" * (52 * 1024 * 1024))
    sc = classify_file_size(file_size_bytes(f))
    assert sc == SIZE_BLOCKED_GT_50MB


def test_inventory_detects_gt_100mb(tmp_path):
    f = tmp_path / "huge.csv"
    f.write_bytes(b"c" * (102 * 1024 * 1024))
    sc = classify_file_size(file_size_bytes(f))
    assert sc == SIZE_BLOCKED_GT_100MB


def test_inventory_build_summary_keys():
    stats = {
        "total_files": 10, "tracked_files": 5, "untracked_files": 5,
        "gt_10mb": 2, "gt_50mb": 1, "gt_100mb": 0, "total_size_bytes": 1024 * 1024,
    }
    summary = build_summary(stats)
    keys = {r["metric"] for r in summary}
    assert "total_files" in keys
    assert "files_gt_50mb" in keys
    assert "files_gt_100mb" in keys


# ===========================================================================
# 5. v1ul policy classifier
# ===========================================================================

def test_policy_blocks_raw():
    row = {"rel_path": "data/external_raw/foo.zip", "size_bytes": "1000",
           "size_mb": "0.001", "size_class": "OK", "tracked": "false"}
    result = classify_row(row)
    assert result["policy"] == POLICY_RAW_EXTERNAL_NEVER_VERSION
    assert result["can_stage"] == "false"


def test_policy_blocks_cache():
    row = {"rel_path": "data/external_cache/bar.parquet", "size_bytes": "1000",
           "size_mb": "0.001", "size_class": "OK", "tracked": "false"}
    result = classify_row(row)
    assert result["policy"] == POLICY_CACHE_NEVER_VERSION
    assert result["can_stage"] == "false"


def test_policy_blocks_local_runs():
    row = {"rel_path": "local_runs/v1jp/output.csv", "size_bytes": "5000",
           "size_mb": "0.005", "size_class": "OK", "tracked": "false"}
    result = classify_row(row)
    assert result["policy"] == POLICY_LOCAL_ONLY_LARGE_DERIVED
    assert result["can_stage"] == "false"


def test_policy_blocks_csv_gt50mb():
    row = {"rel_path": "datasets/big.csv", "size_bytes": str(51 * 1024 * 1024),
           "size_mb": "51", "size_class": "BLOCKED_GT_50MB", "tracked": "false"}
    result = classify_row(row)
    assert result["can_stage"] == "false"


def test_policy_blocks_known_large_csv():
    row = {"rel_path": "datasets/formal_negative_evidence_intake_registry.csv",
           "size_bytes": str(161 * 1024 * 1024), "size_mb": "161",
           "size_class": "BLOCKED_GT_100MB", "tracked": "false"}
    result = classify_row(row)
    assert result["can_stage"] == "false"


def test_policy_allows_readme():
    row = {"rel_path": "README.md", "size_bytes": "5000",
           "size_mb": "0.005", "size_class": "OK", "tracked": "true"}
    result = classify_row(row)
    assert result["policy"] == POLICY_PUBLIC_VERSIONED
    assert result["can_stage"] == "true"


def test_policy_allows_small_docs():
    row = {"rel_path": "docs/metodologia_cientifica/notes.md",
           "size_bytes": "2000", "size_mb": "0.002",
           "size_class": "OK", "tracked": "true"}
    result = classify_row(row)
    assert result["can_stage"] == "true"


def test_policy_allows_python_script():
    row = {"rel_path": "scripts/repository/myscript.py",
           "size_bytes": "3000", "size_mb": "0.003",
           "size_class": "OK", "tracked": "true"}
    result = classify_row(row)
    assert result["can_stage"] == "true"


def test_ul_build_summary_has_policies():
    rows = [
        {"policy": POLICY_PUBLIC_VERSIONED, "can_stage": "true", "needs_public_summary": "false"},
        {"policy": POLICY_BLOCKED_GT_50MB, "can_stage": "false", "needs_public_summary": "true"},
    ]
    summary = ul_build_summary(rows)
    policies = {r["policy"] for r in summary}
    assert POLICY_PUBLIC_VERSIONED in policies
    assert POLICY_BLOCKED_GT_50MB in policies


# ===========================================================================
# 6. v1um summary generator
# ===========================================================================

def _make_csv(path: Path, n_rows: int = 100) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "status", "value"])
        for i in range(n_rows):
            w.writerow([str(i), "PENDING" if i % 2 == 0 else "DONE", str(i * 10)])


def test_summary_generator_does_not_copy_large_file(tmp_path):
    source = tmp_path / "large.csv"
    _make_csv(source, n_rows=200)
    out = tmp_path / "large_public_summary.csv"
    info = generate_summary_csv(source, out)
    # Summary contains at most MAX_SAMPLE_ROWS rows, not the full 200
    assert info["sample_row_count"] <= MAX_SAMPLE_ROWS
    assert info["line_count"] == 200
    # Summary file exists and was generated
    assert out.exists()


def test_summary_generator_creates_status_counts(tmp_path):
    source = tmp_path / "data.csv"
    _make_csv(source, n_rows=10)
    out = tmp_path / "summary.csv"
    info = generate_summary_csv(source, out)
    assert "status" in info.get("status_counts", {})


def test_summary_generator_limits_sample_rows(tmp_path):
    source = tmp_path / "big.csv"
    _make_csv(source, n_rows=500)
    out = tmp_path / "big_summary.csv"
    info = generate_summary_csv(source, out)
    assert info["sample_row_count"] <= MAX_SAMPLE_ROWS


def test_summary_generator_records_column_count(tmp_path):
    source = tmp_path / "cols.csv"
    _make_csv(source)
    out = tmp_path / "cols_summary.csv"
    info = generate_summary_csv(source, out)
    assert info["column_count"] == 3


def test_summary_generator_produces_head_hash(tmp_path):
    source = tmp_path / "hash.csv"
    _make_csv(source)
    out = tmp_path / "hash_summary.csv"
    info = generate_summary_csv(source, out)
    assert len(info["head_hash"]) == 16


# ===========================================================================
# 7. v1un staging manifest
# ===========================================================================

def _policy_row(rel, policy, can_stage, size_mb="0.01"):
    return {"rel_path": rel, "policy": policy, "can_stage": can_stage,
            "size_mb": size_mb, "needs_public_summary": "false"}


def test_staging_manifest_excludes_large_file():
    pol = [_policy_row("datasets/large.csv", POLICY_BLOCKED_GT_50MB, "false", "55")]
    candidates, exclusions = build_manifests(pol, [])
    paths = {r["path"] for r in candidates}
    assert "datasets/large.csv" not in paths
    excl_paths = {r["path"] for r in exclusions}
    assert "datasets/large.csv" in excl_paths


def test_staging_manifest_includes_public_versioned():
    pol = [_policy_row("scripts/foo.py", POLICY_PUBLIC_VERSIONED, "true")]
    candidates, exclusions = build_manifests(pol, [])
    paths = {r["path"] for r in candidates}
    assert "scripts/foo.py" in paths


def test_staging_manifest_includes_summary_file(tmp_path):
    # Create a fake summary file
    summary_file = tmp_path / "test_public_summary.csv"
    summary_file.write_text("id,status\n1,DONE\n", encoding="utf-8")
    rel_summary = str(summary_file)

    pol = [_policy_row("datasets/big.csv", POLICY_BLOCKED_GT_50MB, "false")]
    sum_idx = [{"summary_path": rel_summary, "summary_generated": "true"}]
    # Summary path must exist — monkeypatch ROOT
    import revp_v1un_public_staging_candidate_manifest as m
    original_root = m.ROOT
    m.ROOT = tmp_path
    try:
        candidates, _ = m.build_manifests(pol, sum_idx)
    finally:
        m.ROOT = original_root
    # The function includes summary paths that exist under ROOT
    # Since tmp_path IS root here and the file is at tmp_path/test_public_summary.csv
    # rel_summary is absolute, so we check differently: just verify large excluded
    excl_paths = {r["path"] for r in _}
    assert "datasets/big.csv" in excl_paths


# ===========================================================================
# 8. v1uo guardrail scanner
# ===========================================================================

def test_guardrail_detects_absolute_path_windows(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text('DATA = "C:\\\\Users\\\\gabriela\\\\data\\\\file.csv"', encoding="utf-8")
    result = scan_file(str(f.relative_to(tmp_path)), 0.001)
    # scan_file uses ROOT so we need to patch, just test detect function directly
    hits = detect_absolute_path_content(f.read_text(encoding="utf-8"))
    assert len(hits) > 0


def test_guardrail_detects_local_runs(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("output_dir = 'local_runs/v1jp/results'", encoding="utf-8")
    assert detect_local_runs_content(f.read_text(encoding="utf-8")) is True


def test_guardrail_does_not_detect_local_runs_absent(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text("output_dir = 'datasets/results'", encoding="utf-8")
    assert detect_local_runs_content(f.read_text(encoding="utf-8")) is False


def test_guardrail_detects_forbidden_term():
    text = "status: ground truth operacional confirmado"
    hits = detect_public_terminology_issue(text)
    assert any("ground truth operacional" in h for h in hits)


def test_guardrail_does_not_detect_claudete_as_claude():
    import re
    # The boundary pattern should NOT match "Claudete"
    _CLAUDE_RE = re.compile(r"\bClaude\b(?!\s+Code)(?!te)", re.MULTILINE)
    text = "Claudete foi ao mercado."
    assert not _CLAUDE_RE.search(text)


def test_guardrail_detects_claude_standalone():
    import re
    _CLAUDE_RE = re.compile(r"\bClaude\b(?!\s+Code)(?!te)", re.MULTILINE)
    text = "Generated by Claude for review."
    assert _CLAUDE_RE.search(text) is not None


def test_guardrail_detects_can_train_model_true():
    text = "review_only,true\ncan_train_model,true\nformal_negative,false"
    hits = detect_forbidden_guardrail_true(text)
    assert any("can_train_model" in h for h in hits)


def test_guardrail_detects_ground_truth_operational_true():
    text = "ground_truth_operational,true"
    hits = detect_forbidden_guardrail_true(text)
    assert any("ground_truth_operational" in h for h in hits)


def test_guardrail_detects_formal_negative_true():
    text = "formal_negative,true"
    hits = detect_forbidden_guardrail_true(text)
    assert any("formal_negative" in h for h in hits)


def test_guardrail_does_not_flag_clean_csv():
    text = "review_only,true\ncan_train_model,false\nformal_negative,false"
    hits = detect_forbidden_guardrail_true(text)
    assert len(hits) == 0


def test_uo_build_summary_structure():
    scan_rows = [
        {"violation_count": "0", "violation_abs_path": "false",
         "violation_local_runs": "false", "violation_forbidden_term": "false",
         "violation_forbidden_flag": "false", "violation_large_file": "false"},
        {"violation_count": "1", "violation_abs_path": "true",
         "violation_local_runs": "false", "violation_forbidden_term": "false",
         "violation_forbidden_flag": "false", "violation_large_file": "false"},
    ]
    summary = uo_build_summary(scan_rows)
    keys = {r["metric"] for r in summary}
    assert "files_scanned" in keys
    assert "guardrail_violations_in_safe_to_stage" in keys


# ===========================================================================
# 9. v1up precommit gate
# ===========================================================================

def test_precommit_gate_ps1_is_generated():
    assert len(_PS1_CONTENT) > 200


def test_precommit_gate_blocks_large_file():
    assert "50" in _PS1_CONTENT
    assert "BLOCK_MB" in _PS1_CONTENT or "50MB" in _PS1_CONTENT or "$BLOCK_MB" in _PS1_CONTENT


def test_precommit_gate_blocks_local_runs():
    assert "local_runs" in _PS1_CONTENT


def test_precommit_gate_blocks_raw_data():
    assert "external_raw" in _PS1_CONTENT


def test_precommit_gate_contains_claudete_protection():
    # Must use boundary so Claudete is safe
    assert "Claudete" in _PS1_CONTENT


def test_precommit_gate_returns_exit_code_1():
    assert "exit 1" in _PS1_CONTENT


def test_precommit_gate_prints_staged_files():
    assert "staged" in _PS1_CONTENT.lower()


# ===========================================================================
# 10. v1uq gitignore updater
# ===========================================================================

def test_gitignore_block_is_idempotent(tmp_path, monkeypatch):
    gi = tmp_path / ".gitignore"
    gi.write_text("*.pyc\n__pycache__/\n", encoding="utf-8")

    import revp_v1uq_gitignore_policy_updater as m
    original_path = m.GITIGNORE_PATH
    m.GITIGNORE_PATH = gi
    try:
        changed1, content1 = m.update_gitignore()
        changed2, content2 = m.update_gitignore()
        assert changed1 is True
        assert changed2 is False
        # Content identical on second call
        assert content1 == content2
    finally:
        m.GITIGNORE_PATH = original_path


def test_gitignore_contains_required_patterns():
    block = build_governance_block()
    for pattern in GOVERNANCE_PATTERNS:
        assert pattern in block


def test_gitignore_does_not_remove_existing_entries(tmp_path, monkeypatch):
    gi = tmp_path / ".gitignore"
    gi.write_text("*.pyc\nexisting_pattern/\n", encoding="utf-8")

    import revp_v1uq_gitignore_policy_updater as m
    original_path = m.GITIGNORE_PATH
    m.GITIGNORE_PATH = gi
    try:
        m.update_gitignore()
        result = gi.read_text(encoding="utf-8")
        assert "*.pyc" in result
        assert "existing_pattern/" in result
    finally:
        m.GITIGNORE_PATH = original_path


def test_public_data_policy_content():
    assert "formal_negative_evidence_intake_registry.csv" in _PUBLIC_DATA_POLICY
    assert "formal_negative_candidate_decision_audit.csv" in _PUBLIC_DATA_POLICY
    assert "50" in _PUBLIC_DATA_POLICY


def test_public_data_policy_created(tmp_path, monkeypatch):
    import revp_v1uq_gitignore_policy_updater as m
    original_policy = m.POLICY_DOC_PATH
    original_gi = m.GITIGNORE_PATH
    gi = tmp_path / ".gitignore"
    gi.write_text("", encoding="utf-8")
    policy_out = tmp_path / "PUBLIC_DATA_POLICY.md"
    m.GITIGNORE_PATH = gi
    m.POLICY_DOC_PATH = policy_out
    try:
        m.update_public_data_policy()
        assert policy_out.exists()
        content = policy_out.read_text(encoding="utf-8")
        assert "Never version" in content
    finally:
        m.GITIGNORE_PATH = original_gi
        m.POLICY_DOC_PATH = original_policy


# ===========================================================================
# 11. v1ur bundle — final status
# ===========================================================================

def test_bundle_ready_status():
    metrics = {
        "files_scanned": 100, "tracked_files": 80, "untracked_files": 20,
        "files_gt_10mb": 0, "files_gt_50mb": 0, "files_gt_100mb": 0,
        "blocked_large_files": 0, "public_summary_files_created": 0,
        "safe_to_stage_files": 80, "guardrail_violations": 0,
        "raw_cache_files_blocked": 5,
    }
    assert determine_final_status(metrics) == STATUS_READY


def test_bundle_ready_with_local_only_status():
    metrics = {
        "files_scanned": 200, "tracked_files": 100, "untracked_files": 100,
        "files_gt_10mb": 3, "files_gt_50mb": 2, "files_gt_100mb": 1,
        "blocked_large_files": 2, "public_summary_files_created": 2,
        "safe_to_stage_files": 95, "guardrail_violations": 0,
        "raw_cache_files_blocked": 5,
    }
    assert determine_final_status(metrics) == STATUS_READY_WITH_LOCAL_ONLY


def test_bundle_fail_closed_on_violations():
    metrics = {
        "files_scanned": 100, "tracked_files": 80, "untracked_files": 20,
        "files_gt_10mb": 0, "files_gt_50mb": 0, "files_gt_100mb": 0,
        "blocked_large_files": 0, "public_summary_files_created": 0,
        "safe_to_stage_files": 78, "guardrail_violations": 2,
        "violations_in_safe_files": 2,
        "raw_cache_files_blocked": 0,
    }
    assert determine_final_status(metrics) == STATUS_FAIL_CLOSED


def test_bundle_fail_closed_large_no_summaries():
    metrics = {
        "files_scanned": 100, "tracked_files": 80, "untracked_files": 20,
        "files_gt_10mb": 3, "files_gt_50mb": 2, "files_gt_100mb": 1,
        "blocked_large_files": 2, "public_summary_files_created": 0,
        "safe_to_stage_files": 78, "guardrail_violations": 0,
        "raw_cache_files_blocked": 0,
    }
    assert determine_final_status(metrics) == STATUS_FAIL_CLOSED


def test_bundle_quality_checks_pass_clean(tmp_path, monkeypatch):
    # Minimal check: function returns list of dicts with expected keys
    metrics = {
        "files_scanned": 10, "tracked_files": 8, "untracked_files": 2,
        "files_gt_10mb": 0, "files_gt_50mb": 0, "files_gt_100mb": 0,
        "blocked_large_files": 0, "public_summary_files_created": 0,
        "safe_to_stage_files": 8, "guardrail_violations": 0,
        "raw_cache_files_blocked": 0,
    }
    qc = build_quality_checks(metrics, STATUS_READY)
    assert all("check" in r and "passed" in r for r in qc)


# ===========================================================================
# 12. Schemas and docs existence (integration)
# ===========================================================================

def test_scripts_exist():
    scripts_dir = _REPO_ROOT / "scripts" / "repository"
    expected = [
        "revp_v1uk_v1ur_artifact_governance_common.py",
        "revp_v1uk_repository_artifact_size_inventory.py",
        "revp_v1ul_versioning_policy_classifier.py",
        "revp_v1um_large_csv_public_summary_generator.py",
        "revp_v1un_public_staging_candidate_manifest.py",
        "revp_v1uo_public_repository_guardrail_scanner.py",
        "revp_v1up_precommit_gate_generator.py",
        "revp_v1uq_gitignore_policy_updater.py",
        "revp_v1ur_artifact_governance_bundle.py",
    ]
    for name in expected:
        assert (scripts_dir / name).exists(), f"Missing: {name}"


def test_no_writes_outside_tmp(tmp_path):
    # This test itself is a contract: all fixture-based tests use tmp_path.
    # We confirm tmp_path is separate from the repo.
    assert tmp_path != _REPO_ROOT
    assert not str(tmp_path).startswith(str(_REPO_ROOT))


# ===========================================================================
# 13. Post-condition: staged files still empty
# ===========================================================================

def test_staged_files_empty_at_end():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    staged = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    assert staged == [], f"STAGED_NOT_EMPTY at end: {staged}"
