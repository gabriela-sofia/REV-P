from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ground_truth"))

from revp_v2ez_to_v2ff_comum import GLOBAL_LIMITS, read_csv, table


def test_v2ez_to_v2ff_integrated_outputs_exist() -> None:
    resumo = read_csv(ROOT / "outputs_public" / "logs_summary" / "revp_v2ez_to_v2ff_resumo_testes.csv")
    assert len(resumo) >= 8


def test_v2ez_to_v2ff_command_line() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/ground_truth/revp_v2ez_to_v2ff_orquestrador.py", "--force"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0
    assert '"ground_truth_operational_status": "ABSENT"' in result.stdout


def test_v2ez_to_v2ff_relatorio_integrado_exists() -> None:
    assert (ROOT / "outputs_public" / "execution_reports" / "revp_v2ez_to_v2ff_relatorio_integrado.md").exists()


def test_v2ez_to_v2ff_resumo_limites_preserva_estado_global() -> None:
    rows = read_csv(ROOT / "outputs_public" / "logs_summary" / "revp_v2ez_to_v2ff_resumo_limites.csv")
    for key, value in GLOBAL_LIMITS.items():
        assert any(row["limite_metodologico"] == key and row["valor"] == value for row in rows)


def test_v2ez_to_v2ff_painel_no_training_ready() -> None:
    row = read_csv(table(ROOT, "revp_painel_perda_recuperacao_base_original_v2ff.csv"))[0]
    assert row["ground_truth_operational_status"] == "ABSENT"
    assert row["original_53_recoverable"] in {"true", "false"}


RESUMO_ROWS = read_csv(ROOT / "outputs_public" / "logs_summary" / "revp_v2ez_to_v2ff_resumo_testes.csv")


@pytest.mark.parametrize("row", RESUMO_ROWS)
def test_v2ez_to_v2ff_resumo_rows_pass(row: dict[str, str]) -> None:
    assert row["status"] == "PASS"
    assert "auditoria forense somente leitura" in row["resumo_bloqueio"]




