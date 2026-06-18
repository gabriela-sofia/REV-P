from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2ez_to_v2ff_comum import EXPECTED_FILES, GLOBAL_LIMITS, read_csv, run_integrated, table


def ensure_outputs() -> None:
    if not table(ROOT, "revp_painel_perda_recuperacao_base_original_v2ff.csv").exists():
        run_integrated(ROOT, True)


ensure_outputs()
ROWS = read_csv(table(ROOT, "revp_indice_busca_forense_repositorio_v2ez.csv"))
GUARDS = read_csv(ROOT / "outputs_public" / "logs_summary" / "revp_limites_busca_forense_repositorio_v2ez.csv")


def test_v2ez_search_index_exists() -> None:
    assert ROWS


def test_v2ez_search_is_forensic_only() -> None:
    assert all("COPY" not in row["forensic_status"] for row in ROWS)
    assert all("RESTORE" not in row["forensic_status"] for row in ROWS)


def test_v2ez_expected_exact_names_are_conservative() -> None:
    exact_names = {row["file_name"] for row in ROWS if row["is_exact_expected_artifact_name"] == "true"}
    assert exact_names <= set(EXPECTED_FILES)


def test_v2ez_resumo_limites_preserva_ausencia() -> None:
    for key, value in GLOBAL_LIMITS.items():
        assert any(row["limite_metodologico"] == key and row["valor"] == value for row in GUARDS)


@pytest.mark.parametrize("row", ROWS[:40])
def test_v2ez_rows_have_non_operational_claim(row: dict[str, str]) -> None:
    assert "auditoria forense somente para revisao" in row["allowed_claim"]
    assert row["forbidden_claim"]


@pytest.mark.parametrize("row", ROWS[:40])
def test_v2ez_rows_do_not_mark_training(row: dict[str, str]) -> None:
    assert "liberacao para treinamento" in row["forbidden_claim"]
    assert row["forensic_status"].startswith("FORENSIC_")



