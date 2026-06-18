from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2es_to_v2ey_common import read_csv, run_v2eu, table

run_v2eu(ROOT, True, False)
DRY = read_csv(table(ROOT, "revp_controlled_artifact_recovery_manifest_v2eu.csv"))
run_v2eu(ROOT, True, True)
APPROVED = read_csv(table(ROOT, "revp_controlled_artifact_recovery_manifest_v2eu.csv"))


def test_v2eu_manifest_exists() -> None:
    assert DRY
    assert APPROVED


@pytest.mark.parametrize("row", DRY[:80])
def test_v2eu_dry_run_copies_nothing(row: dict[str, str]) -> None:
    assert row["recover_approved"] == "false"
    assert row["copy_performed"] == "false"


@pytest.mark.parametrize("row", APPROVED[:80])
def test_v2eu_approved_only_copies_valid_candidates(row: dict[str, str]) -> None:
    if row["copy_performed"] == "true":
        assert row["source_sha256"] == row["destination_sha256"]
    else:
        assert row["copy_status"] in {"COPY_SKIPPED_INVALID_CANDIDATE", "COPY_SKIPPED_NO_CANDIDATES"}


def test_v2eu_no_unapproved_copy_status_in_approved_run() -> None:
    assert all(row["copy_status"] != "COPY_SKIPPED_DRY_RUN" for row in APPROVED)

