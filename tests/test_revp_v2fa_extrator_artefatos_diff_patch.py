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
ROWS = read_csv(table(ROOT, "revp_candidatos_artefatos_diff_patch_v2fa.csv"))
MANIFEST = read_csv(table(ROOT, "revp_manifesto_arquivos_embutidos_diff_patch_v2fa.csv"))


def test_v2fa_diff_candidate_table_exists() -> None:
    assert ROWS


def test_v2fa_does_not_apply_or_extract() -> None:
    assert all(row["extraction_status"].startswith("DIFF_") for row in ROWS)
    assert all(row["blocking_reason"] != "patch applied" for row in ROWS)


def test_v2fa_manifest_is_readonly_when_present() -> None:
    assert all(row["future_action"] in {"revisao manual de extracao", "sem extracao"} for row in MANIFEST)


@pytest.mark.parametrize("row", ROWS[:35])
def test_v2fa_candidates_keep_extraction_explicit(row: dict[str, str]) -> None:
    assert row["extraction_possible"] in {"true", "false"}
    assert row["contains_full_file_content"] in {"true", "false"}


@pytest.mark.parametrize("row", ROWS[:35])
def test_v2fa_candidates_have_limite_metodologico_language(row: dict[str, str]) -> None:
    assert "auditoria forense somente para revisao" in row["allowed_claim"]
    assert row["forbidden_claim"]



