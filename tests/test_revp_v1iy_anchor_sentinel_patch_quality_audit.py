"""
Tests for REV-P v1iy local Sentinel patch quality audit.

The tests enforce that local quality metadata can upgrade review readiness, but
never creates labels, training permission, Protocol B reopening, or public raw
raster artifacts.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1iy_anchor_sentinel_patch_quality_audit.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iy"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"

RUN_CMD = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-v1ix-manifest",
    "--audit-local-patches",
    "--compute-spectral-qa",
    "--emit-quality-decision",
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
        "v1iy_patch_quality_inventory.csv",
        "v1iy_band_statistics.csv",
        "v1iy_spectral_index_preview_stats.csv",
        "v1iy_cloud_quality_audit.csv",
        "v1iy_patch_pair_quality_decision.csv",
        "v1iy_summary.json",
        "v1iy_qa.csv",
    ]
    for name in required:
        assert (LOCAL_RUNS / name).exists(), f"Missing local v1iy output: {name}"


def test_public_registry_schema_and_gate_matrix_are_created() -> None:
    run_script_once()
    assert (DATASETS / "official_anchor_sentinel_patch_quality_registry.csv").exists()
    assert (SCHEMAS / "official_anchor_sentinel_patch_quality_schema.csv").exists()
    assert (DATASETS / "official_anchor_patch_pair_quality_gate_matrix.csv").exists()
    assert (SCHEMAS / "official_anchor_patch_pair_quality_gate_matrix_schema.csv").exists()


def test_patch_without_label_training_or_protocol_release() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1iy_summary.json").read_text(encoding="utf-8"))
    registry_rows = read_csv(DATASETS / "official_anchor_sentinel_patch_quality_registry.csv")
    qa_rows = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1iy_qa.csv")}

    assert summary["can_be_operational_ground_truth"] is False
    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False
    assert summary["can_reopen_protocol_b"] is False
    assert qa_rows["can_create_training_label_false"] == "PASS"
    assert qa_rows["can_train_model_false"] == "PASS"
    assert qa_rows["can_reopen_protocol_b_false"] == "PASS"
    for row in registry_rows:
        assert row["can_be_operational_ground_truth"] == "false"
        assert row["can_create_training_label"] == "false"
        assert row["can_train_model"] == "false"
        assert row["can_reopen_protocol_b"] == "false"


def test_high_global_cloud_metadata_does_not_auto_reject_without_local_mask() -> None:
    run_script_once()
    rows = read_csv(DATASETS / "official_anchor_sentinel_patch_quality_registry.csv")
    pre = next(row for row in rows if row["temporal_relation_to_event"] == "PRE_EVENT")

    assert float(pre["cloud_metadata_global"]) >= 80.0
    assert pre["cloud_mask_available"] == "false"
    assert pre["local_cloud_fraction"] == "CLOUD_MASK_NOT_AVAILABLE"
    assert pre["pair_quality_status"] == "PRE_PATCH_CLOUD_RISK_HIGH"
    assert pre["local_quality_status"] == "LOCAL_QA_PASS"
    assert pre["can_be_reference_patch_candidate"] == "true"


def test_spectral_qa_and_gate_matrix_are_consistent() -> None:
    run_script_once()
    index_rows = read_csv(LOCAL_RUNS / "v1iy_spectral_index_preview_stats.csv")
    gate_rows = {row["gate"]: row for row in read_csv(DATASETS / "official_anchor_patch_pair_quality_gate_matrix.csv")}

    assert {row["index_name"] for row in index_rows} == {"NDWI_APPROX_B03_B08", "NDBI_APPROX_B11_B08"}
    assert all(row["index_computable"] == "true" for row in index_rows)
    assert gate_rows["patch_exists"]["status"] == "PASS"
    assert gate_rows["bands_complete"]["status"] == "PASS"
    assert gate_rows["valid_pixels"]["status"] == "PASS"
    assert gate_rows["cloud_local_assessed"]["status"] == "WARN"
    assert gate_rows["spectral_indices_computable"]["status"] == "PASS"


def test_no_private_path_in_public_outputs() -> None:
    run_script_once()
    public_files = [
        DATASETS / "official_anchor_sentinel_patch_quality_registry.csv",
        SCHEMAS / "official_anchor_sentinel_patch_quality_schema.csv",
        DATASETS / "official_anchor_patch_pair_quality_gate_matrix.csv",
        SCHEMAS / "official_anchor_patch_pair_quality_gate_matrix_schema.csv",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


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


def test_docs_explain_quality_and_do_not_use_forbidden_terms() -> None:
    doc_files = [
        DOCS / "protocolo_c_qualidade_patch_sentinel_anchor_v1iy.md",
        DOCS / "protocolo_c_relatorio_qualidade_patch_sentinel_anchor_v1iy.md",
    ]
    for path in doc_files:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "cloudy_pixel_percentage" in text or "metadado global de nuvem" in text
        assert "patch local" in text
        assert "label" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
