"""Tests for v1nh gazette negative resolution report."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1nh_gazette_negative_resolution_report.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1ng_c4_recheck_after_gazette_negatives.py"
SUMMARY = ROOT / "datasets/protocol_c_gazette_negative_resolution_summary.csv"
DOCS = [
    ROOT / "docs/metodologia_cientifica/protocolo_c_diario_oficial_negativos_v1na_v1nh.md",
    ROOT / "docs/metodologia_cientifica/protocolo_c_relatorio_diario_oficial_negativos_v1na_v1nh.md",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-summary"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert SUMMARY.exists() and all(path.exists() for path in DOCS)


def test_summary_keeps_c4_blocked_without_formal_negative() -> None:
    row = rows(SUMMARY)[0]
    if row["formal_negative_count"] == "0":
        assert row["c4_decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
        assert row["can_create_operational_label"] == "false"
    assert row["can_train_model"] == "false"


def test_public_report_has_no_private_paths_or_raw_artifacts() -> None:
    text = SUMMARY.read_text(encoding="utf-8") + "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    low = text.lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert ".zip" not in low and ".shp" not in low and ".npy" not in low and ".npz" not in low
