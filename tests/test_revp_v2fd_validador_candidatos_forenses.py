from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2ez_to_v2ff_comum import read_csv, run_integrated, table


def ensure_outputs() -> None:
    if not table(ROOT, "revp_painel_perda_recuperacao_base_original_v2ff.csv").exists():
        run_integrated(ROOT, True)


ensure_outputs()
ROWS = read_csv(table(ROOT, "revp_validacao_candidatos_forenses_v2fd.csv"))
COUNTS = read_csv(table(ROOT, "revp_comparacao_contagens_candidatos_forenses_v2fd.csv"))


def test_v2fd_validation_exists() -> None:
    assert ROWS


def test_v2fd_expected_count_is_53() -> None:
    assert all(row["expected_reference_rows"] == "53" for row in ROWS)


def test_v2fd_count_comparison_exists() -> None:
    assert COUNTS and COUNTS[0]["expected_events"] == "53"


@pytest.mark.parametrize("row", ROWS[:40])
def test_v2fd_no_positive_or_negative_gate_promoted(row: dict[str, str]) -> None:
    assert row["has_positive_gate_closed"] in {"true", "false"}
    assert row["has_negative_gate_closed"] == "false"
    assert row["ground_truth_operational_status"] in {"ABSENT", "NON_ABSENT"}


@pytest.mark.parametrize("row", ROWS[:40])
def test_v2fd_invalid_candidates_are_not_restored(row: dict[str, str]) -> None:
    if row["validation_status"] != "FORENSIC_CANDIDATE_VALID_RECOVERY_SOURCE":
        assert row["recovery_recommendation"] != "controlled restore candidate"

