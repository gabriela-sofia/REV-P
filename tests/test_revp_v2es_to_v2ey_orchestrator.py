from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2es_to_v2ey_common import GLOBAL_GUARDS, read_csv, run_integrated, table


def test_v2es_to_v2ey_dry_run_complete() -> None:
    outputs = run_integrated(ROOT, True, False)
    assert len(outputs) >= 30


def test_v2es_to_v2ey_recover_approved_complete() -> None:
    outputs = run_integrated(ROOT, True, True)
    assert len(outputs) >= 30


def test_v2es_to_v2ey_reports_exist() -> None:
    assert (ROOT / "outputs_public" / "execution_reports" / "revp_v2es_to_v2ey_integrated_report.md").exists()
    assert (ROOT / "outputs_public" / "execution_reports" / "revp_v2es_to_v2ey_scientific_summary.md").exists()


def test_v2es_to_v2ey_guardrails_preserved() -> None:
    rows = read_csv(ROOT / "outputs_public" / "logs_summary" / "revp_v2es_to_v2ey_guardrail_rollup.csv")
    for key, value in GLOBAL_GUARDS.items():
        assert any(row["guardrail"] == key and row["value"] == value for row in rows)


def test_v2es_to_v2ey_command_line_dry_run() -> None:
    result = subprocess.run([sys.executable, "scripts/ground_truth/revp_v2es_to_v2ey_orchestrator.py", "--force"], cwd=ROOT, check=False, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0
    assert '"ground_truth_operational_status": "ABSENT"' in result.stdout


def test_v2es_to_v2ey_command_line_recover_approved() -> None:
    result = subprocess.run([sys.executable, "scripts/ground_truth/revp_v2es_to_v2ey_orchestrator.py", "--recover-approved", "--force"], cwd=ROOT, check=False, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0
    assert '"training_ready": false' in result.stdout


def test_v2es_to_v2ey_final_dashboard_no_labels() -> None:
    row = read_csv(table(ROOT, "revp_ground_truth_recovery_final_dashboard_v2ey.csv"))[0]
    assert row["ground_truth_operational_status"] == "ABSENT"

