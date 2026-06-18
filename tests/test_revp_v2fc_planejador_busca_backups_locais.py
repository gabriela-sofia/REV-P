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
ROWS = read_csv(table(ROOT, "revp_plano_busca_backups_locais_v2fc.csv"))


def test_v2fc_backup_plan_exists() -> None:
    assert len(ROWS) >= 6


def test_v2fc_remote_is_manual_reference() -> None:
    remote = [row for row in ROWS if "GitHub remoto" in row["candidate_location"]]
    assert remote
    assert remote[0]["requires_manual_action"] == "true"


def test_v2fc_no_download_action() -> None:
    assert all("download" not in row["search_status"].lower() for row in ROWS)


@pytest.mark.parametrize("row", ROWS)
def test_v2fc_search_flags_are_boolean_text(row: dict[str, str]) -> None:
    assert row["location_exists"] in {"true", "false"}
    assert row["search_performed"] in {"true", "false"}
    assert row["requires_manual_action"] in {"true", "false"}


@pytest.mark.parametrize("row", ROWS)
def test_v2fc_recommendations_are_non_destructive(row: dict[str, str]) -> None:
    assert "nao baixar neste fluxo" in row["recommended_manual_action"]


