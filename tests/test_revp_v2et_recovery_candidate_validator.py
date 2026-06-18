from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2es_to_v2ey_common import EXPECTED, read_csv, run_v2et, table

run_v2et(ROOT, True)
ROWS = read_csv(table(ROOT, "revp_recovery_candidate_validation_v2et.csv"))
SCHEMA = read_csv(table(ROOT, "revp_recovery_candidate_schema_matrix_v2et.csv"))


def test_v2et_validation_exists() -> None:
    assert ROWS


def test_v2et_schema_matrix_exists_or_no_core_candidates() -> None:
    core = [row for row in ROWS if row["artifact_role"] in EXPECTED]
    assert SCHEMA or not core


@pytest.mark.parametrize("row", ROWS[:80])
def test_v2et_candidate_status_is_explicit(row: dict[str, str]) -> None:
    assert row["candidate_status"].startswith("CANDIDATE_")


@pytest.mark.parametrize("row", ROWS[:80])
def test_v2et_rejects_out_of_scope_without_copy(row: dict[str, str]) -> None:
    if row["artifact_role"] not in EXPECTED:
        assert row["candidate_status"] == "CANDIDATE_REJECTED_OUT_OF_SCOPE"


@pytest.mark.parametrize("row", ROWS[:80])
def test_v2et_hash_is_recorded(row: dict[str, str]) -> None:
    if row["source_path"]:
        assert row["source_sha256"] == "" or len(row["source_sha256"]) == 64

