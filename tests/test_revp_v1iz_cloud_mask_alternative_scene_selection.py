"""
Tests for REV-P v1iz cloud mask and alternative Sentinel scene selection.

The tests enforce controlled cloud-mask handling, pre-event-only alternatives,
metadata-only public outputs, and the no-label/no-training invariants.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1iz_cloud_mask_alternative_scene_selection.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iz"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"

RUN_CMD = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-v1ix-v1iy",
    "--download-cloud-masks",
    "--search-alternative-pre-scenes",
    "--evaluate-local-cloud",
    "--emit-final-pair-decision",
]

PRIVATE_FRAGMENTS = [
    r"C:\Users\gabriela",
    "gabriela",
    r"Documents\REV-P",
    "Documents/REV-P",
    r"Documents\PROJETO",
    "Documents/PROJETO",
]

FORBIDDEN_DOC_TERMS = [
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


def test_script_runs_and_emits_local_outputs() -> None:
    result = run_script_once()
    assert result.returncode == 0, result.stderr
    required = [
        "v1iz_cloud_mask_download_log.csv",
        "v1iz_local_cloud_quality_audit.csv",
        "v1iz_alternative_pre_scene_search.csv",
        "v1iz_alternative_patch_quality_audit.csv",
        "v1iz_final_patch_pair_selection.csv",
        "v1iz_summary.json",
        "v1iz_qa.csv",
    ]
    for name in required:
        assert (LOCAL_RUNS / name).exists(), f"Missing local v1iz output: {name}"


def test_scl_qa60_when_available_computes_local_cloud() -> None:
    run_script_once()
    rows = read_csv(LOCAL_RUNS / "v1iz_local_cloud_quality_audit.csv")
    available = [row for row in rows if row["cloud_mask_available"] == "true"]
    if available:
        assert all(row["cloud_mask_source"] == "SCL_QA60" for row in available)
        assert all(row["local_cloud_fraction"] != "" for row in available)
    else:
        assert all(row["mask_quality_status"] == "QA_MASK_NOT_AVAILABLE" for row in rows)


def test_alternative_pre_scenes_never_cross_event_date() -> None:
    run_script_once()
    rows = read_csv(LOCAL_RUNS / "v1iz_alternative_patch_quality_audit.csv")
    assert rows
    assert all(row["pre_event_valid"] == "true" for row in rows)
    assert all(row["scene_date"] < "2022-02-15" for row in rows)


def test_public_registry_exists_for_useful_decision_and_has_invariants() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1iz_summary.json").read_text(encoding="utf-8"))
    registry = DATASETS / "official_anchor_sentinel_patch_pair_selection_registry.csv"
    schema = SCHEMAS / "official_anchor_sentinel_patch_pair_selection_schema.csv"

    if summary["status"] in {
        "PATCH_PAIR_USABLE_FOR_REVIEW",
        "PRE_PATCH_CLOUD_RISK_RESOLVED",
        "PRE_PATCH_CLOUD_RISK_REMAINS",
        "S2_PRE_EVENT_CLEAR_PATCH_NOT_AVAILABLE",
        "PATCH_PAIR_BLOCKED_BY_LOCAL_CLOUD",
        "QA_MASK_NOT_AVAILABLE_BUT_LOCAL_SPECTRAL_QA_PASS",
    }:
        assert registry.exists()
        assert schema.exists()
        rows = read_csv(registry)
        assert rows
        row = rows[0]
        assert row["can_create_training_label"] == "false"
        assert row["can_train_model"] == "false"
        assert row["can_reopen_protocol_b"] == "false"
        assert row["can_be_operational_ground_truth"] == "false"


def test_can_create_training_label_and_train_model_false_always() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1iz_summary.json").read_text(encoding="utf-8"))
    qa_rows = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1iz_qa.csv")}

    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False
    assert summary["can_reopen_protocol_b"] is False
    assert summary["can_be_operational_ground_truth"] is False
    assert qa_rows["can_create_training_label_false"] == "PASS"
    assert qa_rows["can_train_model_false"] == "PASS"
    assert qa_rows["can_reopen_protocol_b_false"] == "PASS"


def test_raster_artifacts_are_not_versioned() -> None:
    run_script_once()
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(REVP_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    bad_lines = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if any(lowered.endswith(ext) for ext in [".tif", ".tiff", ".npy", ".npz"]):
            if "local_runs/" not in lowered and r"local_runs\\" not in lowered:
                bad_lines.append(line)
    assert not bad_lines, f"Raster or array artifacts are visible to git: {bad_lines}"


def test_no_private_path_in_public_outputs() -> None:
    run_script_once()
    public_files = [
        DATASETS / "official_anchor_sentinel_patch_pair_selection_registry.csv",
        SCHEMAS / "official_anchor_sentinel_patch_pair_selection_schema.csv",
    ]
    for path in public_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_script_once()
    doc_files = [
        DOCS / "protocolo_c_selecao_par_sentinel_anchor_v1iz.md",
        DOCS / "protocolo_c_relatorio_selecao_par_sentinel_anchor_v1iz.md",
    ]
    for path in doc_files:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "scl/qa60" in text
        assert "patch_pair_usable_for_review" in text
        assert "label" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
