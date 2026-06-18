from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2es_to_v2ey_common import EXPECTED, read_csv, run_v2ev, table

run_v2ev(ROOT, True)
ROWS = read_csv(table(ROOT, "revp_recovered_base_verification_v2ev.csv"))
SUMMARY = read_csv(table(ROOT, "revp_recovered_base_count_summary_v2ev.csv"))[0]


def test_v2ev_verifies_seven_core_tables() -> None:
    assert len(ROWS) == len(EXPECTED)


@pytest.mark.parametrize("row", ROWS)
def test_v2ev_status_is_blocked_or_verified(row: dict[str, str]) -> None:
    assert row["verification_status"] in {"VERIFIED_REVIEW_ONLY", "VERIFICATION_BLOCKED"}


def test_v2ev_summary_counts_present() -> None:
    assert SUMMARY["events_count"].isdigit()
    assert SUMMARY["packets_count"].isdigit()
    assert SUMMARY["review_items_count"].isdigit()


def test_v2ev_no_positive_or_negative_gates() -> None:
    assert SUMMARY["positive_gate_closed_count"] == "0"
    assert SUMMARY["negative_gate_closed_count"] == "0"

