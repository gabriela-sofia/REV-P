"""Tests for v1nn DINO review-only triage."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nn_dino_c3_control_triage_queue.py"
ANCHORS = ROOT / "datasets/dino_c3_anchor_review_triage_registry.csv"
CONTROLS = ROOT / "datasets/dino_control_candidate_review_queue.csv"
BOUNDARY = ROOT / "datasets/dino_embedding_training_boundary_matrix.csv"
SCHEMA = ROOT / "datasets/schemas/dino_embedding_training_boundary_schema.csv"
PUBLIC = [ANCHORS, CONTROLS, BOUNDARY, SCHEMA, ROOT / "docs/metodologia_cientifica/protocolo_c_dino_triagem_review_only_v1nn.md"]
ABS_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1nn_dino_stays_review_only() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    boundary = rows(BOUNDARY)[0]
    assert boundary["embedding_can_create_label"] == "false"
    assert boundary["embedding_can_validate_event"] == "false"
    assert boundary["embedding_can_prioritize_review"] == "true"
    assert boundary["embedding_can_support_negative_search_queue"] == "true"
    assert boundary["embedding_can_train_classifier_now"] == "false"
    assert boundary["training_blocker"] == "FORMAL_NEGATIVES_ZERO;C4_NOT_OPEN;SPLIT_LEAKAGE_NOT_READY"


def test_v1nn_anchor_and_control_outputs_do_not_create_labels() -> None:
    assert rows(ANCHORS)
    assert rows(CONTROLS)
    assert {row["can_create_label"] for row in rows(ANCHORS)} == {"false"}
    assert {row["can_create_label"] for row in rows(CONTROLS)} == {"false"}
    assert {row["can_train_model"] for row in rows(CONTROLS)} == {"false"}
    for path in PUBLIC:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not ABS_PATH.search(text)
        assert ".npy" not in text and ".npz" not in text


def test_v1nn_schema_and_determinism() -> None:
    schema_fields = {row["field"] for row in rows(SCHEMA)}
    assert "embedding_can_train_classifier_now" in schema_fields
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    before = {path: digest(path) for path in PUBLIC}
    subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    assert before == {path: digest(path) for path in PUBLIC}
