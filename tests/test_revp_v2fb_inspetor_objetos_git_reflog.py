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
ROWS = read_csv(table(ROOT, "revp_artefatos_objetos_git_reflog_v2fb.csv"))


def test_v2fb_git_inspection_exists() -> None:
    assert ROWS


def test_v2fb_is_readonly_inspection() -> None:
    assert all("RESTORED" not in row["inspection_status"] for row in ROWS)


def test_v2fb_recoverable_flag_is_boolean_text() -> None:
    assert all(row["recoverable_from_git"] in {"true", "false"} for row in ROWS)


@pytest.mark.parametrize("row", ROWS[:35])
def test_v2fb_rows_have_git_source_type(row: dict[str, str]) -> None:
    assert row["git_source_type"] in {"log", "reflog", "fsck", "none"}


@pytest.mark.parametrize("row", ROWS[:35])
def test_v2fb_rows_keep_recovery_conditional(row: dict[str, str]) -> None:
    if row["recoverable_from_git"] == "true":
        assert row["inspection_status"] == "GIT_BLOB_CONTENT_MATCH"
    else:
        assert row["inspection_status"].startswith("GIT_")

