from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2es_to_v2ey_common import read_csv, run_v2ew, table


def test_v2ew_execution_validation_exists() -> None:
    run_v2ew(ROOT, True)
    rows = read_csv(table(ROOT, "revp_restored_v2dz_to_v2ef_execution_validation_v2ew.csv"))
    assert len(rows) == 1


def test_v2ew_ground_truth_absent() -> None:
    row = read_csv(table(ROOT, "revp_restored_v2dz_to_v2ef_execution_validation_v2ew.csv"))[0]
    assert row["ground_truth_operational_status"] == "ABSENT"


def test_v2ew_status_explicit() -> None:
    row = read_csv(table(ROOT, "revp_restored_v2dz_to_v2ef_execution_validation_v2ew.csv"))[0]
    assert row["execution_validation_status"] in {"RESTORED_V2DZ_TO_V2EF_VALID_REVIEW_ONLY", "RESTORED_EXECUTION_BLOCKED_OR_INCOMPLETE"}

