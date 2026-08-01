"""Testes reais (sintéticos, não network) pra mineração de negativos SIAC 156 -- susc_20k3.
Réplica do critério de negativo do SEDEC/Recife (5 condições), sem copiar a lista de categorias.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/scripts"))
import negative_categories_curitiba as cat_mod  # noqa: E402
import mine_negative_candidates_siac156 as mine_mod  # noqa: E402


# --- negative_categories_curitiba -----------------------------------------------------------

def test_known_negative_category_recognized():
    assert cat_mod.is_curitiba_negative_category("Iluminação Pública", "Manutenção de Luminárias") is True


def test_flood_category_never_in_negative_list():
    assert cat_mod.is_curitiba_negative_category("Drenagem", "Alagamentos") is False
    assert cat_mod.is_curitiba_negative_category("Proteção ao cidadão - GM", "Enchentes, inundações ou alagamentos") is False


def test_rain_linked_categories_are_excluded_not_included():
    # Drenagem e Pavimentacao e queda de arvore NAO podem estar na allowlist de negativo
    excluded_pairs = {(a, s) for a, s, _ in cat_mod.CURITIBA_EXCLUDED_RAIN_LINKED if s != "*"}
    included_pairs = {(a, s) for a, s, _ in cat_mod.CURITIBA_NEG_CATEGORIES}
    assert excluded_pairs.isdisjoint(included_pairs)


def test_no_duplicate_categories_in_allowlist():
    pairs = [(a, s) for a, s, _ in cat_mod.CURITIBA_NEG_CATEGORIES]
    assert len(pairs) == len(set(pairs))


def test_every_category_has_real_justification_text():
    for assunto, subdivisao, justificativa in cat_mod.CURITIBA_NEG_CATEGORIES:
        assert len(justificativa) > 20  # nao pode ser vazio/generico demais


# --- mine_negative_candidates_siac156 --------------------------------------------------------

def test_stable_rank_is_deterministic():
    r1 = mine_mod.stable_rank("a", "b", "c")
    r2 = mine_mod.stable_rank("a", "b", "c")
    assert r1 == r2


def test_stable_rank_differs_for_different_input():
    assert mine_mod.stable_rank("a") != mine_mod.stable_rank("b")


def test_load_positive_bairros_normalizes(tmp_path):
    p = tmp_path / "pos.csv"
    p.write_text("bairro,other\nXaxim,x\nRebouças,y\n", encoding="utf-8")
    bairros = mine_mod.load_positive_bairros(p)
    assert bairros == {"XAXIM", "REBOUCAS"}


def test_mine_year_file_applies_category_and_bairro_pairing(tmp_path):
    csv_path = tmp_path / "siac.csv"
    fieldnames = ["Tipo", "Orgao", "DataCriacao", "Assunto", "Subdivisao", "Situacao", "Logradouro", "Bairro", "Regional", "DataResposta", "Origem"]
    import csv as csv_mod
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        # negativo valido: categoria ok + bairro com positivo
        w.writerow({"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "01/01/2025", "Assunto": "Iluminação Pública", "Subdivisao": "Manutenção de Luminárias", "Situacao": "Aberto", "Logradouro": "Rua A", "Bairro": "XAXIM", "Regional": "", "DataResposta": "", "Origem": "Mobile"})
        # categoria nao esta na allowlist -- deve ser descartado
        w.writerow({"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "01/01/2025", "Assunto": "Drenagem", "Subdivisao": "Alagamentos", "Situacao": "Aberto", "Logradouro": "Rua B", "Bairro": "XAXIM", "Regional": "", "DataResposta": "", "Origem": "Mobile"})
        # categoria ok mas bairro sem positivo -- deve ser descartado (condicao 5)
        w.writerow({"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "01/01/2025", "Assunto": "Iluminação Pública", "Subdivisao": "Manutenção de Luminárias", "Situacao": "Aberto", "Logradouro": "Rua C", "Bairro": "BAIRRO_SEM_POSITIVO", "Regional": "", "DataResposta": "", "Origem": "Mobile"})

    rows = mine_mod.mine_year_file(csv_path, positive_bairros={"XAXIM"}, max_por_ano=None)
    assert len(rows) == 1
    assert rows[0]["bairro"] == "XAXIM"
    assert rows[0]["assunto"] == "Iluminação Pública"


def test_mine_year_file_respects_max_por_ano(tmp_path):
    csv_path = tmp_path / "siac.csv"
    fieldnames = ["Tipo", "Orgao", "DataCriacao", "Assunto", "Subdivisao", "Situacao", "Logradouro", "Bairro", "Regional", "DataResposta", "Origem"]
    import csv as csv_mod
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for i in range(10):
            w.writerow({"Tipo": "Solicitação", "Orgao": "X", "DataCriacao": "01/01/2025", "Assunto": "Iluminação Pública", "Subdivisao": "Manutenção de Luminárias", "Situacao": "Aberto", "Logradouro": f"Rua {i}", "Bairro": "XAXIM", "Regional": "", "DataResposta": "", "Origem": "Mobile"})

    rows = mine_mod.mine_year_file(csv_path, positive_bairros={"XAXIM"}, max_por_ano=3)
    assert len(rows) == 3
