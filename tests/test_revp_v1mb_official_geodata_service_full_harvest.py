"""Tests for v1mb official geodata service harvest."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mb_official_geodata_service_full_harvest.py"
REG = ROOT / "datasets/official_geodata_service_registry.csv"
LAYERS = ROOT / "datasets/official_geodata_layer_inventory.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and LAYERS.exists()


def test_layers_are_classified_without_ground_truth_promotion() -> None:
    data = rows(LAYERS)
    assert data
    assert all(r["gt_potential"] in {"HIGH_GT_POTENTIAL", "MODERATE_GT_REVIEW", "CONTEXT_ONLY"} for r in data)
    assert "FORMAL_NEGATIVE" not in LAYERS.read_text(encoding="utf-8")


def test_public_outputs_have_no_private_paths() -> None:
    low = (REG.read_text(encoding="utf-8") + LAYERS.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
