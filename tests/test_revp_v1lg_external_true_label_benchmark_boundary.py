"""Tests for v1lg external benchmark boundary."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lg_external_true_label_benchmark_boundary.py"
OUT = ROOT / "datasets/external_true_label_benchmark_registry.csv"
BOUNDARY = ROOT / "datasets/external_benchmark_transfer_boundary_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-boundary"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and BOUNDARY.exists()


def test_external_benchmark_never_transfers_ground_truth_to_petropolis() -> None:
    reg = rows(OUT)[0]
    assert reg["can_transfer_ground_truth_to_petropolis"] == "false"
    assert "formal negative" in reg["forbidden_use"]
