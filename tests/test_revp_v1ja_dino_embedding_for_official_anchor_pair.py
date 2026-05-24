"""
Tests for REV-P v1ja DINO embeddings for the official anchor Sentinel pair.

The tests allow a controlled DINO_MODEL_UNAVAILABLE blocker, but if embeddings
are generated they must be 768D, finite, review-only, and non-supervised.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1ja_dino_embedding_for_official_anchor_pair.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ja"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"

RUN_CMD = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-v1iz-selection",
    "--read-local-patches",
    "--load-dinov2",
    "--extract-embeddings",
    "--emit-structural-diagnostics",
    "--emit-readiness",
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
        "v1ja_anchor_pair_input_audit.csv",
        "v1ja_dino_embedding_manifest_local.csv",
        "v1ja_embedding_diagnostics.csv",
        "v1ja_structural_pair_comparison.csv",
        "v1ja_summary.json",
        "v1ja_qa.csv",
    ]
    for name in required:
        assert (LOCAL_RUNS / name).exists(), f"Missing local v1ja output: {name}"


def test_dino_unavailable_is_controlled_or_embeddings_are_768d() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1ja_summary.json").read_text(encoding="utf-8"))

    if summary["status"] == "DINO_MODEL_UNAVAILABLE":
        assert summary["primary_blocker"] == "DINO_MODEL_UNAVAILABLE"
        assert summary["embedding_generated_pre"] is False
        assert summary["embedding_generated_post"] is False
    else:
        assert summary["status"] == "DINO_ANCHOR_PAIR_EMBEDDING_READY"
        assert summary["embedding_dim"] == 768
        assert summary["embedding_generated_pre"] is True
        assert summary["embedding_generated_post"] is True


def test_generated_embeddings_are_finite_and_pair_metrics_exist() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1ja_summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "DINO_ANCHOR_PAIR_EMBEDDING_READY":
        return
    diagnostics = read_csv(LOCAL_RUNS / "v1ja_embedding_diagnostics.csv")
    comparison = read_csv(LOCAL_RUNS / "v1ja_structural_pair_comparison.csv")[0]

    assert len(diagnostics) == 2
    assert all(row["embedding_dim"] == "768" for row in diagnostics)
    assert all(row["has_nan"] == "false" for row in diagnostics)
    assert all(row["has_inf"] == "false" for row in diagnostics)
    assert float(comparison["cosine_similarity"]) >= -1.0
    assert float(comparison["cosine_similarity"]) <= 1.0
    assert float(comparison["euclidean_distance"]) >= 0.0


def test_public_registry_only_exists_after_embedding_qa_pass() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1ja_summary.json").read_text(encoding="utf-8"))
    registry = DATASETS / "official_anchor_dino_embedding_readiness_registry.csv"
    schema = SCHEMAS / "official_anchor_dino_embedding_readiness_schema.csv"

    if summary["status"] == "DINO_ANCHOR_PAIR_EMBEDDING_READY":
        assert registry.exists()
        assert schema.exists()
        rows = read_csv(registry)
        assert rows
        assert all(row["embedding_quality_status"] == "QA_PASS" for row in rows)
        assert all(row["can_be_review_embedding"] == "true" for row in rows)
    else:
        assert not registry.exists()


def test_dino_does_not_create_label_or_training_permission() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1ja_summary.json").read_text(encoding="utf-8"))
    qa_rows = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1ja_qa.csv")}
    manifest_rows = read_csv(LOCAL_RUNS / "v1ja_dino_embedding_manifest_local.csv")

    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False
    assert summary["can_reopen_protocol_b"] is False
    assert summary["can_be_operational_ground_truth"] is False
    assert qa_rows["can_create_training_label_false"] == "PASS"
    assert qa_rows["can_train_model_false"] == "PASS"
    assert qa_rows["can_reopen_protocol_b_false"] == "PASS"
    for row in manifest_rows:
        assert row["can_create_training_label"] == "false"
        assert row["can_train_model"] == "false"
        assert row["can_reopen_protocol_b"] == "false"
        assert row["can_be_operational_ground_truth"] == "false"


def test_multimodal_readiness_matrix_is_emitted() -> None:
    run_script_once()
    matrix = DATASETS / "official_anchor_multimodal_reference_readiness_matrix.csv"
    schema = SCHEMAS / "official_anchor_multimodal_reference_readiness_schema.csv"
    assert matrix.exists()
    assert schema.exists()
    rows = {row["gate"]: row for row in read_csv(matrix)}
    assert rows["official_anchor"]["status"] == "PASS"
    assert rows["sentinel_pair_selected"]["status"] == "PASS"
    assert rows["dino_embedding_generated"]["status"] in {"PASS", "FAIL"}


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


def test_no_private_path_in_public_outputs() -> None:
    run_script_once()
    public_files = [
        DATASETS / "official_anchor_dino_embedding_readiness_registry.csv",
        SCHEMAS / "official_anchor_dino_embedding_readiness_schema.csv",
        DATASETS / "official_anchor_multimodal_reference_readiness_matrix.csv",
        SCHEMAS / "official_anchor_multimodal_reference_readiness_schema.csv",
    ]
    for path in public_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_detection_prediction_terms() -> None:
    run_script_once()
    doc_files = [
        DOCS / "protocolo_c_embedding_dino_anchor_oficial_v1ja.md",
        DOCS / "protocolo_c_relatorio_embedding_dino_anchor_oficial_v1ja.md",
    ]
    for path in doc_files:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "frozen" in text
        assert "embedding" in text
        assert "label" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
