"""
Tests for REV-P v1jb official-anchor multimodal change probe.

The tests enforce that spectral and frozen-DINO diagnostics can support a
review-only probe, while training, labels, unfreeze claims, and Protocol B remain
blocked.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1jb_anchor_multimodal_change_probe.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jb"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"

RUN_CMD = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-v1iz-selection",
    "--read-v1iy-quality",
    "--read-v1ja-dino",
    "--compute-spectral-deltas",
    "--compare-reference-distribution",
    "--evaluate-training-boundary",
    "--optional-unfrozen-sandbox",
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
    "detecÃ§Ã£o",
    "prediÃ§Ã£o",
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
        "v1jb_anchor_multimodal_feature_delta.csv",
        "v1jb_dino_spectral_change_probe.csv",
        "v1jb_reference_distribution_comparison.csv",
        "v1jb_training_boundary_decision.csv",
        "v1jb_unfrozen_sandbox_log.csv",
        "v1jb_summary.json",
        "v1jb_qa.csv",
    ]
    for name in required:
        assert (LOCAL_RUNS / name).exists(), f"Missing local v1jb output: {name}"


def test_registry_schema_created_for_useful_probe() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1jb_summary.json").read_text(encoding="utf-8"))
    registry = DATASETS / "official_anchor_multimodal_change_probe_registry.csv"
    schema = SCHEMAS / "official_anchor_multimodal_change_probe_schema.csv"

    if summary["status"] == "MULTIMODAL_CHANGE_PROBE_READY":
        assert registry.exists()
        assert schema.exists()
        rows = read_csv(registry)
        assert rows
        assert rows[0]["can_be_multimodal_reference_candidate"] == "true"


def test_ndwi_ndbi_deltas_are_calculated() -> None:
    run_script_once()
    rows = {row["feature"]: row for row in read_csv(LOCAL_RUNS / "v1jb_anchor_multimodal_feature_delta.csv")}
    summary = json.loads((LOCAL_RUNS / "v1jb_summary.json").read_text(encoding="utf-8"))

    assert "NDWI_mean" in rows
    assert "NDBI_mean" in rows
    assert rows["NDWI_mean"]["delta"] == summary["ndwi_delta"]
    assert rows["NDBI_mean"]["delta"] == summary["ndbi_delta"]
    assert float(summary["ndwi_delta"]) != 0.0
    assert float(summary["ndbi_delta"]) != 0.0


def test_dino_distance_and_structural_status_are_present() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1jb_summary.json").read_text(encoding="utf-8"))
    probe = read_csv(LOCAL_RUNS / "v1jb_dino_spectral_change_probe.csv")[0]

    assert float(summary["dino_euclidean_distance"]) > 0.0
    assert float(probe["dino_euclidean_distance"]) == float(summary["dino_euclidean_distance"])
    assert probe["structural_change_status"].endswith("_REVIEW_ONLY")
    assert summary["structural_change_status"].endswith("_REVIEW_ONLY")


def test_reference_distribution_status_is_controlled() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1jb_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(LOCAL_RUNS / "v1jb_reference_distribution_comparison.csv")

    assert rows
    assert summary["reference_distribution_status"] in {
        "REFERENCE_DISTRIBUTION_INSUFFICIENT",
        "DINO_REFERENCE_AVAILABLE",
        "SPECTRAL_REFERENCE_LIMITED",
        "DINO_REFERENCE_AVAILABLE+SPECTRAL_REFERENCE_LIMITED",
    }
    assert all(row["comparison_status"] for row in rows)


def test_training_and_unfreeze_remain_blocked() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1jb_summary.json").read_text(encoding="utf-8"))
    registry_row = read_csv(DATASETS / "official_anchor_multimodal_change_probe_registry.csv")[0]
    boundary_rows = read_csv(LOCAL_RUNS / "v1jb_training_boundary_decision.csv")
    qa_rows = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1jb_qa.csv")}

    assert summary["training_boundary_status"] == "TRAINING_BLOCKED_INSUFFICIENT_LABELS"
    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False
    assert summary["can_unfreeze_model_for_scientific_claim"] is False
    assert summary["can_reopen_protocol_b"] is False
    assert registry_row["can_create_training_label"] == "false"
    assert registry_row["can_train_model"] == "false"
    assert registry_row["can_unfreeze_model_for_scientific_claim"] == "false"
    assert registry_row["can_reopen_protocol_b"] == "false"
    assert all(row["can_train_model"] == "false" for row in boundary_rows)
    assert all(row["can_unfreeze_model_for_scientific_claim"] == "false" for row in boundary_rows)
    assert qa_rows["training_blocked"] == "PASS"
    assert qa_rows["can_train_model_false"] == "PASS"
    assert qa_rows["can_unfreeze_model_for_scientific_claim_false"] == "PASS"


def test_unfrozen_sandbox_is_invalid_or_skipped_controlled() -> None:
    run_script_once()
    rows = read_csv(LOCAL_RUNS / "v1jb_unfrozen_sandbox_log.csv")
    assert rows
    row = rows[0]
    if row["sandbox_ran"] == "true":
        assert row["sandbox_status"] == "INVALID_FOR_SCIENTIFIC_CLAIM"
    else:
        assert row["sandbox_status"] == "SANDBOX_SKIPPED_BY_SCIENTIFIC_GUARDRAIL"


def test_no_private_path_in_public_outputs() -> None:
    run_script_once()
    public_files = [
        DATASETS / "official_anchor_multimodal_change_probe_registry.csv",
        SCHEMAS / "official_anchor_multimodal_change_probe_schema.csv",
    ]
    for path in public_files:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_raster_npy_npz_artifacts_are_not_versioned() -> None:
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


def test_docs_do_not_use_forbidden_terms() -> None:
    run_script_once()
    doc_files = [
        DOCS / "protocolo_c_probe_multimodal_anchor_oficial_v1jb.md",
        DOCS / "protocolo_c_relatorio_probe_multimodal_anchor_oficial_v1jb.md",
    ]
    for path in doc_files:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "frozen" in text
        assert "label" in text
        assert "training_blocked_insufficient_labels" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
