from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2es_to_v2ey_common import read_csv, run_v2ey, table


def test_v2ey_dashboard_exists() -> None:
    run_v2ey(ROOT, True)
    rows = read_csv(table(ROOT, "revp_ground_truth_recovery_final_dashboard_v2ey.csv"))
    assert len(rows) == 1


def test_v2ey_next_actions_exist() -> None:
    rows = read_csv(table(ROOT, "revp_ground_truth_recovery_next_actions_v2ey.csv"))
    assert rows and rows[0]["next_action"]


def test_v2ey_final_status_explicit() -> None:
    row = read_csv(table(ROOT, "revp_ground_truth_recovery_final_dashboard_v2ey.csv"))[0]
    assert row["recovery_status"] in {"RECOVERY_COMPLETE_53_RESTORED_REVIEW_ONLY", "RECOVERY_FALLBACK_ONLY_38_REVIEW_ONLY", "RECOVERY_PARTIAL_COUNTS_DIFFER_REVIEW_ONLY", "RECOVERY_BLOCKED_NO_VALID_SOURCE"}


def test_v2ey_ground_truth_absent() -> None:
    row = read_csv(table(ROOT, "revp_ground_truth_recovery_final_dashboard_v2ey.csv"))[0]
    assert row["ground_truth_operational_status"] == "ABSENT"

