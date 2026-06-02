"""Tests for v1ms C4 final recheck after feature exhaustion."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ms_c4_final_recheck_after_feature_exhaust.py"
DEP1 = ROOT / "scripts/protocolo_c/revp_v1mq_patch_plan_new_formal_negative.py"
DEP2 = ROOT / "scripts/protocolo_c/revp_v1mr_real_c4_split_leakage_recheck.py"
OUT = ROOT / "datasets/c4_final_recheck_after_feature_exhaust.csv"
READY = ROOT / "datasets/c4_label_readiness_after_feature_exhaust.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP1), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    subprocess.run([sys.executable, str(DEP2), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and READY.exists()


def test_c4_only_opens_with_formal_negative_patch_and_split() -> None:
    row = rows(OUT)[0]
    if row["formal_negative_count"] == "0":
        assert row["decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
        assert row["can_create_operational_label"] == "false"
    assert row["can_train_model"] == "false"


def test_public_outputs_are_sanitized() -> None:
    low = OUT.read_text(encoding="utf-8").lower() + READY.read_text(encoding="utf-8").lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
