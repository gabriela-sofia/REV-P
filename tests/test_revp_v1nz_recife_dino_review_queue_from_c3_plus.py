"""Tests for v1nz Recife DINO C3+ review queue."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nz_recife_dino_review_queue_from_c3_plus.py"
OUT = ROOT / "datasets/recife_dino_c3_plus_review_queue.csv"
NEIGHBOR = ROOT / "datasets/recife_dino_c3_plus_neighbor_audit_queue.csv"
BOUNDARY = ROOT / "datasets/recife_dino_c3_plus_training_boundary_matrix.csv"


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1nz_dino_queue_is_review_only_and_does_not_create_labels(tmp_path: Path) -> None:
    adj = tmp_path / "adj.csv"
    write_rows(adj, [{"candidate_id": "C1", "patch_id": "REC_PATCH_UNKNOWN", "adjudication_status": "C4_PREFLIGHT_POSITIVE_ONLY_BLOCKED_NO_NEGATIVE", "temporal_class": "TEMPORAL_STRONG"}])
    env = os.environ.copy()
    env["REVP_RECIFE_C3_PLUS_ADJUDICATION_REGISTRY"] = str(adj)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    review = read_rows(OUT)[0]
    assert review["dino_status"] == "REVIEW_ONLY_REPRESENTATION"
    assert review["can_create_label"] == "false"
    assert review["can_validate_event"] == "false"
    assert read_rows(NEIGHBOR)[0]["inherits_label"] == "false"
    boundary = read_rows(BOUNDARY)[0]
    assert boundary["dino_can_prioritize_review"] == "true"
    assert boundary["dino_can_create_label"] == "false"
    assert boundary["dino_can_train_model"] == "false"
