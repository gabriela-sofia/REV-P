"""Tests for v1ly one-page blocker map."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ly_one_page_blocker_map.py"
OUT = ROOT / "datasets/ground_truth_blocker_map_v1ly.csv"
NEXT = ROOT / "datasets/ground_truth_next_action_matrix_v1ly.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and NEXT.exists()


def test_blocker_map_is_minimal_and_has_single_actions() -> None:
    data = rows(OUT)
    assert {r["gate"] for r in data} >= {"formal_positive_gate", "formal_negative_gate", "c4_operational_gate"}
    assert all(r["single_technical_action"] for r in data)
    assert rows(NEXT)[0]["can_train_model"] == "false"


def test_public_outputs_have_no_private_paths() -> None:
    low = (OUT.read_text(encoding="utf-8") + NEXT.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
