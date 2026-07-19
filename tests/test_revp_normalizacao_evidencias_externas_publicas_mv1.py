# -*- coding: utf-8 -*-
"""Testes da normalizacao de evidencias externas publicas (MV1)."""
from __future__ import annotations

import csv
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TBL = os.path.join(REPO, "outputs_public", "tables")
REP = os.path.join(REPO, "outputs_public", "execution_reports")
MET = os.path.join(REPO, "outputs_public", "metrics")

MANIFESTO_PUBLICO = os.path.join(TBL, "revp_manifesto_publico_arquivos_externos_url_hash_mv1.csv")
SUBSTITUICOES = os.path.join(TBL, "revp_normalizacao_bloqueadores_evidencias_externas_mv1.csv")
MAN_NORM = os.path.join(TBL, "revp_manifesto_evidencias_externas_normalizado_mv1.csv")
AUD_NORM = os.path.join(TBL, "revp_auditoria_fontes_externas_normalizada_mv1.csv")
REC_NORM = os.path.join(TBL, "revp_recomendacoes_pacote_externo_normalizada_mv1.csv")
RELATORIO = os.path.join(REP, "revp_normalizacao_evidencias_externas_publicas_mv1.md")
METRICAS = os.path.join(MET, "revp_normalizacao_evidencias_externas_publicas_mv1.json")

# linguagem proibida em campos estruturados (claims e licenca-como-bloqueador)
PROIBIDO_DADOS = [
    "modelo detecta", "modelo prediz", "curitiba e negativo", "classe 0",
    "positivo confirmado", "negativo confirmado", "ground truth fechado",
    "treino liberado", "validacao operacional", "validação operacional",
    "bloqueado por licenca", "bloqueada por licenca", "bloqueado por licença",
    "bloqueada por licença", "licenca como bloqueador", "licença como bloqueador",
    "bloquear_para_uso_publico_ate_confirmar_licenca",
]
TOKEN_BLOQUEIO_LICENCA = "bloquear_para_uso_publico_ate_confirmar_licenca"


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _all_text_lower(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().lower()


def test_todos_artefatos_existem():
    for p in [MANIFESTO_PUBLICO, SUBSTITUICOES, MAN_NORM, AUD_NORM, REC_NORM, RELATORIO, METRICAS]:
        assert os.path.exists(p), f"artefato ausente: {p}"


def test_json_valido_e_valores_obrigatorios():
    d = json.load(open(METRICAS, encoding="utf-8"))
    assert d["total_arquivos_manifesto_publico"] == 8
    assert d["arquivos_com_hash_real"] == 8
    assert d["arquivos_parseaveis"] == 8
    assert d["arquivos_com_url_final"] == 8
    assert d["brutos_pesados_em_outputs_public"] == 0
    assert d["pode_verificar_por_url_hash_tamanho"] is True
    assert d["pode_usar_git_add_A"] is False
    assert d["ground_truth_operacional_status"] == "ausente"
    assert d["treino_supervisionado_status"] == "bloqueado"
    assert d["itens_podem_virar_label_agora"] == 0
    assert d["total_ocorrencias_licenca_como_bloqueador_encontradas"] >= 1
    assert d["total_ocorrencias_normalizadas"] == d["total_ocorrencias_licenca_como_bloqueador_encontradas"]
    assert isinstance(d["guardrails_preservados"], list) and len(d["guardrails_preservados"]) >= 8


def test_manifesto_publico_colunas_obrigatorias():
    rows = _rows(MANIFESTO_PUBLICO)
    obrig = {"id_arquivo", "id_fonte", "instituicao", "cidade", "regiao", "titulo_arquivo",
             "tipo_arquivo", "formato", "url_origem", "url_final_download", "caminho_local_relativo",
             "tamanho_bytes", "sha256", "status_url", "status_hash", "parseabilidade",
             "origem_publica_institucional", "pode_ir_para_git", "motivo_nao_ir_para_git",
             "pode_ir_para_pacote_externo", "papel_metodologico", "observacao"}
    assert obrig.issubset(set(rows[0].keys()))


def test_manifesto_publico_8_arquivos_verificados():
    rows = _rows(MANIFESTO_PUBLICO)
    assert len(rows) == 8
    for r in rows:
        assert len(r["sha256"]) == 64, f"sha256 vazio/invalido em {r['id_arquivo']}"
        assert str(r["tamanho_bytes"]).strip().isdigit(), f"tamanho vazio em {r['id_arquivo']}"
        assert r["parseabilidade"], f"parseabilidade vazia em {r['id_arquivo']}"
        assert r["status_hash"] == "confere", f"hash nao confere em {r['id_arquivo']}"
        assert r["pode_ir_para_git"] == "false"
        assert r["origem_publica_institucional"] == "sim"
        # motivo nao-git deve ser tecnico, nunca licenca
        assert "licen" not in r["motivo_nao_ir_para_git"].lower()


def test_substituicoes_colunas_e_normalizado_sem_token_de_bloqueio():
    rows = _rows(SUBSTITUICOES)
    obrig = {"artefato", "campo_ou_trecho", "valor_antigo", "valor_normalizado",
             "tipo_normalizacao", "justificativa", "status", "observacao"}
    assert obrig.issubset(set(rows[0].keys()))
    assert len(rows) >= 1
    # valor_antigo PODE conter o token (e o registro do que foi removido);
    # valor_normalizado NUNCA pode conter o bloqueador de licenca.
    for r in rows:
        assert TOKEN_BLOQUEIO_LICENCA not in r["valor_normalizado"]
        assert "licen" not in r["valor_normalizado"].lower()


def test_brutos_pesados_em_outputs_public_zero():
    op = os.path.join(REPO, "outputs_public")
    heavy = 0
    for dp, _, files in os.walk(op):
        for f in files:
            fp = os.path.join(dp, f)
            ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
            if ext in {"tif", "tiff", "geotiff", "npz", "npy", "zip", "shp", "gpkg", "img"} or os.path.getsize(fp) > 5_000_000:
                heavy += 1
    assert heavy == 0


def test_artefatos_dados_normalizados_sem_linguagem_proibida():
    for p in [MANIFESTO_PUBLICO, MAN_NORM, AUD_NORM, REC_NORM]:
        txt = _all_text_lower(p)
        for termo in PROIBIDO_DADOS:
            assert termo.lower() not in txt, f"linguagem proibida '{termo}' em {os.path.basename(p)}"


def test_auditoria_normalizada_sem_criterio_de_licenca():
    rows = _rows(AUD_NORM)
    for r in rows:
        assert r["criterio"] != "licenca_ou_uso_claro"
        assert "licen" not in r["acao_recomendada"].lower()
        assert "licen" not in r["risco"].lower()


def test_relatorio_contem_guardrails():
    txt = _all_text_lower(RELATORIO)
    assert "guardrails preservados" in txt
    assert "evidência externa não vira label" in txt
    assert "não declarar ground truth operacional" in txt
