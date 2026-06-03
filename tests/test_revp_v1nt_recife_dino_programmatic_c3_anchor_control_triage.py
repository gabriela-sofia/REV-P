"""Tests for v1nt Recife DINO-programmatic review-only triage."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nt_recife_dino_programmatic_c3_anchor_control_triage.py"
BOUNDARY = ROOT / "datasets/recife_dino_protocol_c_boundary_matrix.csv"
ANCHOR = ROOT / "datasets/recife_dino_c3_anchor_candidate_queue.csv"
NEIGHBOR = ROOT / "datasets/recife_dino_structural_neighbor_review_queue.csv"
DISCORD = ROOT / "datasets/recife_dino_discordance_review_queue.csv"
CONTROL = ROOT / "datasets/recife_dino_control_candidate_queue.csv"
SCHEMA = ROOT / "datasets/schemas/recife_dino_protocol_c_boundary_schema.csv"
DOC = ROOT / "docs/metodologia_cientifica/protocolo_c_recife_dino_triagem_v1nt.md"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_v1nt_dino_boundary_is_review_only() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    boundary = rows(BOUNDARY)[0]
    assert boundary["dino_can_prioritize_review"] == "true"
    assert boundary["dino_can_create_label"] == "false"
    assert boundary["dino_can_validate_event"] == "false"
    assert boundary["dino_can_train_model"] == "false"
    assert boundary["dino_status"] == "REVIEW_ONLY_REPRESENTATION"


def test_v1nt_outputs_do_not_create_labels_or_controls_by_absence() -> None:
    assert "dino_can_train_model" in {row["field"] for row in rows(SCHEMA)}
    for path in [ANCHOR, NEIGHBOR, DISCORD, CONTROL, BOUNDARY, DOC]:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "can_create_label,true" not in text
        assert "dino_can_create_label,true" not in text
        assert "can_be_negative_by_absence,true" not in text
