"""
Tests for REV-P v1iv official anchor Sentinel patch acquisition.

The tests accept the controlled GEE_AUTH_REQUIRED path when Google Earth Engine
is not authenticated. They enforce that no label, target, training run, Protocol
B reopening, or public raster artifact is introduced.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1iv_anchor_sentinel_patch_acquisition.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iv"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs"

RUN_CMD = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--check-gee",
    "--build-gee-export",
    "--try-local-export",
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

FORBIDDEN_DOC_WORDS = [
    "flood detection",
    "landslide detection",
    "flood prediction",
    "landslide prediction",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        "v1iv_gee_availability_check.json",
        "v1iv_sentinel_scene_search.csv",
        "v1iv_selected_scene_decision.csv",
        "v1iv_anchor_patch_manifest_local.csv",
        "v1iv_patch_quality_audit.csv",
        "v1iv_reference_patch_readiness_decision.csv",
        "v1iv_summary.json",
        "v1iv_qa.csv",
        "v1iv_gee_export_plan.js",
        "v1iv_gee_export_plan.py",
    ]
    for name in required:
        assert (LOCAL_RUNS / name).exists(), f"Missing local v1iv output: {name}"


def test_gee_unavailable_status_is_controlled() -> None:
    summary = json.loads((LOCAL_RUNS / "v1iv_summary.json").read_text(encoding="utf-8"))
    availability = json.loads((LOCAL_RUNS / "v1iv_gee_availability_check.json").read_text(encoding="utf-8"))

    if not availability.get("gee_available"):
        assert summary["status"] == "GEE_AUTH_REQUIRED"
        assert summary["primary_blocker"] == "GEE_AUTH_REQUIRED"
        assert summary["patch_generated"] is False
        assert summary["commit_warranted"] is False


def test_export_plans_are_reproducible_and_anchor_specific() -> None:
    js_plan = (LOCAL_RUNS / "v1iv_gee_export_plan.js").read_text(encoding="utf-8")
    py_plan = (LOCAL_RUNS / "v1iv_gee_export_plan.py").read_text(encoding="utf-8")
    combined = js_plan + "\n" + py_plan

    assert "COPERNICUS/S2_SR_HARMONIZED" in combined
    for band in ["B02", "B03", "B04", "B08", "B11", "B12"]:
        assert band in combined
    assert "-22.484251" in combined
    assert "-43.211257" in combined
    assert "2022-02-01" in combined
    assert "2022-03-06" in combined


def test_public_registry_is_not_created_without_real_patch() -> None:
    summary = json.loads((LOCAL_RUNS / "v1iv_summary.json").read_text(encoding="utf-8"))
    registry = DATASETS / "official_anchor_sentinel_patch_registry.csv"

    if not summary.get("patch_generated"):
        if not registry.exists():
            return
        rows = read_csv(registry)
        assert not any(row.get("reference_patch_id", "").startswith("REFPATCH_PET2022_CPRM_MOINHO_PRETO_S2_V1IV") for row in rows)


def test_patch_real_requires_qa_pass_for_public_candidate() -> None:
    summary = json.loads((LOCAL_RUNS / "v1iv_summary.json").read_text(encoding="utf-8"))
    readiness = read_csv(LOCAL_RUNS / "v1iv_reference_patch_readiness_decision.csv")
    assert readiness
    row = readiness[0]

    if summary.get("patch_generated"):
        assert summary["patch_qa_status"] in {"PASS", "FAIL"}
        if summary["patch_qa_status"] == "PASS":
            assert row["can_be_reference_patch_candidate"] == "true"
            assert (DATASETS / "official_anchor_sentinel_patch_registry.csv").exists()
            assert (SCHEMAS / "official_anchor_sentinel_patch_schema.csv").exists()
        else:
            assert row["can_be_reference_patch_candidate"] == "false"


def test_training_and_protocol_invariants_are_always_false() -> None:
    summary = json.loads((LOCAL_RUNS / "v1iv_summary.json").read_text(encoding="utf-8"))
    readiness = read_csv(LOCAL_RUNS / "v1iv_reference_patch_readiness_decision.csv")[0]
    qa_rows = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1iv_qa.csv")}

    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False
    assert summary["can_reopen_protocol_b"] is False
    assert summary["can_be_operational_ground_truth"] is False

    assert readiness["can_create_training_label"] == "false"
    assert readiness["can_train_model"] == "false"
    assert readiness["can_reopen_protocol_b"] == "false"
    assert readiness["can_be_operational_ground_truth"] == "false"
    assert readiness["public_versioning_status"] == "METADATA_ONLY"

    assert qa_rows.get("can_create_training_label_false") == "PASS"
    assert qa_rows.get("can_train_model_false") == "PASS"
    assert qa_rows.get("can_reopen_protocol_b_false") == "PASS"


def test_raster_artifacts_are_not_versioned() -> None:
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


def test_no_private_path_in_public_v1iv_outputs() -> None:
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


def test_v1iv_docs_do_not_use_forbidden_detection_prediction_terms() -> None:
    doc_files = list(DOCS.rglob("*v1iv*.md")) if DOCS.exists() else []
    for path in doc_files:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for word in FORBIDDEN_DOC_WORDS:
            assert word not in text, f"Forbidden wording {word!r} found in {path}"
