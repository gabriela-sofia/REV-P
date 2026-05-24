"""
Tests for REV-P v1ix GEE Sentinel patch export for the official anchor.

The suite accepts controlled GEE_AUTH_REQUIRED or EXPORT_TASK_REQUIRED statuses,
but does not allow labels, training flags, Protocol B reopening, public rasters,
or public metadata without a real QA-passing local patch pair.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1ix_gee_sentinel_patch_export_for_anchor.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ix"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"

RUN_CMD = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--check-gee",
    "--authenticate-check",
    "--search-sentinel2",
    "--export-pre-post-patches",
    "--download-small-patches",
    "--emit-qa",
    "--emit-manifest",
]

PRIVATE_FRAGMENTS = [
    r"C:\Users\gabriela",
    "gabriela",
    r"Documents\REV-P",
    "Documents/REV-P",
    r"Documents\PROJETO",
    "Documents/PROJETO",
]

FORBIDDEN_TERMS = [
    "flood detection",
    "landslide detection",
    "flood prediction",
    "landslide prediction",
    "detecção",
    "predição",
    "predicao",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_script_once() -> subprocess.CompletedProcess[str]:
    return subprocess.run(RUN_CMD, cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)


def test_script_exists_and_compiles() -> None:
    assert SCRIPT.exists()
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(SCRIPT)],
        cwd=str(REVP_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_script_runs_and_emits_required_outputs() -> None:
    result = run_script_once()
    assert result.returncode == 0, result.stderr

    required = [
        "v1ix_gee_environment_check.json",
        "v1ix_sentinel2_scene_search.csv",
        "v1ix_selected_scene_decision.csv",
        "v1ix_patch_download_log.csv",
        "v1ix_anchor_patch_manifest_local.csv",
        "v1ix_patch_quality_audit.csv",
        "v1ix_reference_patch_pair_readiness.csv",
        "v1ix_summary.json",
        "v1ix_qa.csv",
        "v1ix_gee_export_plan.js",
        "v1ix_gee_export_plan.py",
    ]
    for name in required:
        assert (LOCAL_RUNS / name).exists(), f"Missing local v1ix output: {name}"


def test_gee_auth_failure_is_controlled_when_present() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1ix_summary.json").read_text(encoding="utf-8"))
    environment = json.loads((LOCAL_RUNS / "v1ix_gee_environment_check.json").read_text(encoding="utf-8"))

    if not environment.get("gee_available"):
        assert summary["status"] == "GEE_AUTH_REQUIRED"
        assert summary["primary_blocker"] == "GEE_AUTH_REQUIRED"
        assert summary["authentication_instruction"] == "earthengine authenticate"
        assert summary["pre_patch_generated"] is False
        assert summary["post_patch_generated"] is False
        assert summary["commit_warranted"] is False


def test_if_gee_authenticates_scene_search_runs() -> None:
    run_script_once()
    environment = json.loads((LOCAL_RUNS / "v1ix_gee_environment_check.json").read_text(encoding="utf-8"))
    summary = json.loads((LOCAL_RUNS / "v1ix_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(LOCAL_RUNS / "v1ix_sentinel2_scene_search.csv")

    if environment.get("gee_available"):
        assert summary["gee_authenticated"] is True
        assert isinstance(summary["pre_sentinel_scenes_found"], int)
        assert isinstance(summary["post_sentinel_scenes_found"], int)
        assert rows
        assert all(row["gee_collection"] == "COPERNICUS/S2_SR_HARMONIZED" for row in rows)


def test_public_registry_is_not_created_without_real_patch_pair() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1ix_summary.json").read_text(encoding="utf-8"))
    registry = DATASETS / "official_anchor_sentinel_patch_registry.csv"

    if not summary.get("can_be_reference_patch_candidate"):
        if not registry.exists():
            return
        rows = read_csv(registry)
        assert not any("V1IX" in row.get("reference_patch_id", "") for row in rows)


def test_real_patch_pair_requires_qa_pass_for_public_candidate() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1ix_summary.json").read_text(encoding="utf-8"))
    readiness = read_csv(LOCAL_RUNS / "v1ix_reference_patch_pair_readiness.csv")
    assert readiness

    if summary.get("pre_patch_generated") or summary.get("post_patch_generated"):
        assert summary["patch_qa_status"] in {"PASS", "FAIL"}
    if summary.get("can_be_reference_patch_candidate"):
        assert summary["pre_patch_generated"] is True
        assert summary["post_patch_generated"] is True
        assert summary["patch_qa_status"] == "PASS"
        assert all(row["can_be_reference_patch_candidate"] == "true" for row in readiness)
        assert (DATASETS / "official_anchor_sentinel_patch_registry.csv").exists()
        assert (SCHEMAS / "official_anchor_sentinel_patch_schema.csv").exists()


def test_training_and_protocol_invariants_are_always_false() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1ix_summary.json").read_text(encoding="utf-8"))
    readiness = read_csv(LOCAL_RUNS / "v1ix_reference_patch_pair_readiness.csv")
    qa_rows = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1ix_qa.csv")}

    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False
    assert summary["can_reopen_protocol_b"] is False
    assert summary["can_be_operational_ground_truth"] is False

    for row in readiness:
        assert row["can_create_training_label"] == "false"
        assert row["can_train_model"] == "false"
        assert row["can_reopen_protocol_b"] == "false"
        assert row["can_be_operational_ground_truth"] == "false"
        assert row["public_versioning_status"] == "METADATA_ONLY"

    assert qa_rows.get("can_create_training_label_false") == "PASS"
    assert qa_rows.get("can_train_model_false") == "PASS"
    assert qa_rows.get("can_reopen_protocol_b_false") == "PASS"


def test_raster_artifacts_are_not_versioned() -> None:
    run_script_once()
    git_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(REVP_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert git_status.returncode == 0, git_status.stderr
    bad_lines = []
    for line in git_status.stdout.splitlines():
        lowered = line.lower()
        if any(lowered.endswith(ext) for ext in [".tif", ".tiff", ".npy", ".npz"]):
            if "local_runs/" not in lowered and r"local_runs\\" not in lowered:
                bad_lines.append(line)
    assert not bad_lines, f"Raster or array artifacts are visible to git: {bad_lines}"


def test_no_private_path_in_public_v1ix_outputs() -> None:
    run_script_once()
    public_files = [
        DATASETS / "official_anchor_sentinel_patch_registry.csv",
        SCHEMAS / "official_anchor_sentinel_patch_schema.csv",
    ]
    for path in public_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_v1ix_outputs_do_not_use_detection_prediction_terms() -> None:
    run_script_once()
    output_files = [
        path
        for path in LOCAL_RUNS.glob("v1ix_*")
        if path.suffix.lower() in {".csv", ".json", ".js", ".py"}
    ]
    public_files = [
        DATASETS / "official_anchor_sentinel_patch_registry.csv",
        SCHEMAS / "official_anchor_sentinel_patch_schema.csv",
    ]
    for path in output_files + [path for path in public_files if path.exists()]:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for term in FORBIDDEN_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
