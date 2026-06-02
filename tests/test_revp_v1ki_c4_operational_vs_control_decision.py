"""Tests for v1ki C4 operational vs control decision."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/c4_operational_vs_control_decision_matrix.csv"
SUMMARY = ROOT / "datasets/control_experiment_readiness_summary.csv"
sys.path.insert(0, str(SCRIPTS))
from revp_v1ki_c4_operational_vs_control_decision import decide_control_experiment


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    chain = [
        ("revp_v1kf_conservative_stable_control_sampler.py", ["--force", "--emit-controls"]),
        ("revp_v1kg_stable_control_sensitivity_audit.py", ["--force", "--emit-sensitivity"]),
        ("revp_v1kh_positive_stable_control_sandbox_probe.py", ["--force", "--emit-probe"]),
        ("revp_v1ki_c4_operational_vs_control_decision.py", ["--force", "--emit-decision"]),
    ]
    for name, args in chain:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert SUMMARY.exists()


def test_c4_operational_blocked_without_formal_negatives() -> None:
    operational = [row for row in rows(OUT) if row["decision_layer"] == "C4_OPERATIONAL_GROUND_TRUTH"][0]
    assert operational["formal_negative_count"] == "0"
    assert operational["decision"] == "BLOCKED"
    assert operational["can_train_operational_model"] == "false"


def test_control_experiment_ready_only_with_strong_gates() -> None:
    assert decide_control_experiment(9, 12, 9, True) == "C4_CONTROL_EXPERIMENT_READY"
    assert decide_control_experiment(9, 12, 0, True) == "C4_CONTROL_EXPERIMENT_BLOCKED"
    summary = rows(SUMMARY)[0]
    assert summary["c4_control_experiment_status"] == "BLOCKED"


def test_public_outputs_have_no_private_paths() -> None:
    text = OUT.read_text(encoding="utf-8") + SUMMARY.read_text(encoding="utf-8")
    assert "C:\\" not in text and "C:/" not in text
    assert "gabriela" not in text.lower()
