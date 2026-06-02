"""Tests for v1mz C4 recheck after administrative negatives."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mz_c4_recheck_after_administrative_negatives.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1my_formal_negative_administrative_adjudication.py"
C4 = ROOT / "datasets/c4_recheck_after_administrative_negatives.csv"
SUMMARY = ROOT / "datasets/protocol_c_administrative_negative_resolution_summary.csv"
DOCS = [
    ROOT / "docs/metodologia_cientifica/protocolo_c_negativos_administrativos_v1mu_v1mz.md",
    ROOT / "docs/metodologia_cientifica/protocolo_c_relatorio_negativos_administrativos_v1mu_v1mz.md",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-summary"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert C4.exists() and SUMMARY.exists() and all(path.exists() for path in DOCS)


def test_c4_only_opens_with_formal_negative() -> None:
    row = rows(C4)[0]
    if row["formal_negative_count"] == "0":
        assert row["decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
        assert row["can_create_operational_label"] == "false"
    assert row["can_train_model"] == "false"


def test_public_outputs_have_no_private_paths_or_raw_artifacts() -> None:
    text = C4.read_text(encoding="utf-8") + SUMMARY.read_text(encoding="utf-8") + "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    low = text.lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
    assert ".pdf" not in low and ".zip" not in low and ".shp" not in low and ".npy" not in low and ".npz" not in low
