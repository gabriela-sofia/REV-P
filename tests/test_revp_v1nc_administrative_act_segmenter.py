"""Tests for v1nc administrative act segmentation."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DEP = ROOT / "scripts/protocolo_c/revp_v1nb_gazette_full_text_ocr_extraction.py"
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nc_administrative_act_segmenter.py"
SEG = ROOT / "datasets/administrative_act_segment_registry.csv"
HITS = ROOT / "datasets/administrative_act_keyword_hit_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout
    assert SEG.exists() and HITS.exists()


def test_segments_are_not_formal_negative_labels() -> None:
    assert "FORMAL_NEGATIVE_CANDIDATE" not in SEG.read_text(encoding="utf-8")


def test_keyword_hits_are_sanitized() -> None:
    low = HITS.read_text(encoding="utf-8").lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
