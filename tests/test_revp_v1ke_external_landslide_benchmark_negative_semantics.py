"""Tests for v1ke external benchmark semantics."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ke_external_landslide_benchmark_negative_semantics.py"
OUT = ROOT / "datasets/external_landslide_benchmark_semantics_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-registry"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()


def test_landslide4sense_is_external_calibration_not_local_ground_truth() -> None:
    data = {row["benchmark_id"]: row for row in rows(OUT)}
    row = data["LANDSLIDE4SENSE_SEMANTIC_CALIBRATION"]
    assert row["has_non_landslide_labels"] == "true"
    assert row["transfer_status"] == "EXTERNAL_BENCHMARK_FOR_SEMANTIC_CALIBRATION"
    assert row["can_transfer_to_petropolis"] == "false"
    assert "not local ground truth" in row["allowed_use"]


def test_public_registry_has_no_private_path() -> None:
    text = OUT.read_text(encoding="utf-8")
    assert "C:\\" not in text and "C:/" not in text
    assert "gabriela" not in text.lower()
