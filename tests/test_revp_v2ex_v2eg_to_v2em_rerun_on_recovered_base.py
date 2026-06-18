from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2es_to_v2ey_common import read_csv, run_v2ex, table


def test_v2ex_rerun_manifest_exists() -> None:
    run_v2ex(ROOT, True)
    rows = read_csv(table(ROOT, "revp_v2eg_to_v2em_recovered_rerun_v2ex.csv"))
    assert len(rows) == 1


def test_v2ex_rerun_does_not_create_ground_truth() -> None:
    row = read_csv(table(ROOT, "revp_v2eg_to_v2em_recovered_rerun_v2ex.csv"))[0]
    assert row["ground_truth_operational_status"] == "ABSENT"


def test_v2ex_rerun_status_explicit() -> None:
    row = read_csv(table(ROOT, "revp_v2eg_to_v2em_recovered_rerun_v2ex.csv"))[0]
    assert row["rerun_status"] in {"RERUN_SUCCESS_REVIEW_ONLY", "RERUN_SKIPPED_RECOVERED_BASE_INCOMPLETE"}

