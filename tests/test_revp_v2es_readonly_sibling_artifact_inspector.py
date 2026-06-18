from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2es_to_v2ey_common import read_csv, run_v2es, table

run_v2es(ROOT, True)
ROWS = read_csv(table(ROOT, "revp_readonly_sibling_artifact_inspection_v2es.csv"))


def test_v2es_inspection_exists() -> None:
    assert ROWS


def test_v2es_inspects_sibling_worktrees_readonly() -> None:
    assert all(row["inspection_status"] in {"SIBLING_ARTIFACT_FOUND_READONLY", "NO_SIBLING_ARTIFACTS_FOUND"} for row in ROWS)


def test_v2es_records_hashes_for_found_files() -> None:
    found = [row for row in ROWS if row["source_path"]]
    assert found
    assert all(len(row["sha256"]) == 64 for row in found[:20])


@pytest.mark.parametrize("row", ROWS[:80])
def test_v2es_allowed_directory_flag_present(row: dict[str, str]) -> None:
    assert row["allowed_directory"] in {"true", "false"}


@pytest.mark.parametrize("row", ROWS[:80])
def test_v2es_does_not_copy_files(row: dict[str, str]) -> None:
    assert "COPY" not in row["inspection_status"]

