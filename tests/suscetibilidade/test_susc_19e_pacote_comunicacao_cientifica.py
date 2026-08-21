"""Testes do SUSC-19E pacote de comunicacao cientifica review-only.

Cobrem: claim policy com permitidos e proibidos; slides com limitacoes; artigo com
review-only; perguntas de banca com ground truth/previsao/score_v7; Petropolis
bloqueado; score_v7 bloqueado; 17B nao criado; score_v6 intacto; output 19E
versionavel; vocabulario publico proibido falha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "suscetibilidade"
sys.path.insert(0, str(SCRIPTS))

import susc_19e_comunicacao_cientifica_common as s19e  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402


@pytest.fixture(scope="module")
def built():
    summary = s19e.build_all()
    errors = s19e.validate_outputs(summary)
    return summary, errors


def test_validator_sem_erros(built):
    _summary, errors = built
    assert errors == [], f"validator reportou erros: {errors}"


def test_claim_policy_permitidos_e_proibidos(built):
    claims = read_csv(s19e.CLAIMS)
    status = {r["status"] for r in claims}
    assert "permitido" in status
    assert "proibido" in status
    # claims proibidos canonicos presentes
    proib = " ".join(r["frase"].lower() for r in claims if r["status"] == "proibido")
    assert "preve enchentes" in proib
    assert "ground truth patch-level" in proib
    assert "benchmark 17b" in proib
    assert "negativo" in proib
    # nenhum permitido contem padrao perigoso afirmativo
    for r in claims:
        if r["status"] in ("permitido", "permitido_com_ressalva"):
            for _n, pat in s19e.DANGER_PATTERNS.items():
                assert not pat.search(r["frase"]), f"claim permitido perigoso: {r['claim_id']}"


def test_slides_incluem_limitacoes_e_proximos(built):
    txt = s19e.SLIDES.read_text(encoding="utf-8", errors="ignore").lower()
    assert "limitac" in txt
    assert "proximos passos" in txt or "proximo" in txt
    rows = [r for r in s19e.SLIDES.read_text(encoding="utf-8").splitlines() if r.startswith("## Slide")]
    assert len(rows) == 15


def test_artigo_inclui_review_only_e_camadas(built):
    txt = s19e.ART_METODO.read_text(encoding="utf-8", errors="ignore").lower()
    assert "review-only" in txt
    for camada in s19e.SEPARACAO:
        assert camada in txt


def test_banca_cobre_ground_truth_previsao_score_v7(built):
    txt = s19e.BANCA.read_text(encoding="utf-8", errors="ignore").lower()
    assert "ground truth" in txt
    assert "previs" in txt
    assert "score_v7" in txt
    assert "review-only" in txt


def test_petropolis_aparece_bloqueado(built):
    txt = s19e.SINTESE.read_text(encoding="utf-8", errors="ignore").lower()
    assert "petropolis" in txt
    # nunca como observado sem enquadramento
    for path in s19e._prose_paths():
        for line in path.read_text(encoding="utf-8", errors="ignore").lower().splitlines():
            if "petropolis" in line and ("observ" in line or "avaliad" in line):
                import re
                assert re.search(r"(bloque|fora|misto|exclu|nao entra|n[aã]o (e|eh) )", line)


def test_score_v7_bloqueado(built):
    summary, _ = built
    assert summary["status_score_v7"] == "SCORE_V7_NAO_AUTORIZADO"
    assert summary["score_v7_allowed_true_count"] == 0
    assert summary["score_v7_created"] is False


def test_17b_nao_criado(built):
    summary, _ = built
    assert summary["status_17b"] == "17B_NAO_CRIADO"
    assert summary["marco_17b_criado"] is False
    assert summary["benchmark_17b_criado"] is False


def test_score_v6_intacto(built):
    summary, _ = built
    assert summary["score_v6_changed"] is False


def test_missingness_territorial_presente(built):
    joined = " ".join(p.read_text(encoding="utf-8", errors="ignore") for p in s19e._public_paths())
    assert "MapBiomas" in joined
    assert "territorial" in joined.lower()


def test_glossario_completo(built):
    termos = {r["termo"] for r in read_csv(s19e.GLOSSARIO)}
    for req in ("suscetibilidade", "score_v6", "score_v7", "ground truth", "review-only", "SAR", "HAND", "MapBiomas"):
        assert req in termos


def test_riscos_matriz_completa(built):
    riscos = read_csv(s19e.RISCOS)
    assert len(riscos) == 8
    for r in riscos:
        assert r["correcao"].strip()
        assert r["validator_rule"].strip()


def test_plano_figuras(built):
    figs = read_csv(s19e.PLANO_FIGURAS)
    assert len(figs) == 8
    for r in figs:
        assert r["precisa_criar"] in ("true", "false")


def test_output_19e_versionavel(built):
    ignored = s19e.git_ignored([s19e.SUMMARY, s19e.CLAIMS])
    assert s19e.rel(s19e.SUMMARY) not in ignored
    assert s19e.rel(s19e.CLAIMS) not in ignored


def test_vocabulario_publico_proibido_falha():
    for termo in ("agente", "agentic", "codex", "LLM", "IA"):
        assert s19e.public_text_violations_text(f"texto com {termo} embutido")


def test_vocabulario_publico_limpo_nos_outputs(built):
    assert s19e.validate_public_text() == []


def test_schema_gerado(built):
    schema = read_json(s19e.SCHEMA)
    assert schema["guardrails"]["score_v6_changed"]["const"] == "false"
    assert schema["separacao_obrigatoria"] == s19e.SEPARACAO
    assert "19E_COMUNICACAO_PRONTA_COM_RESSALVAS" in schema["status_19e_comunicacao_allowed"]


def test_gates_enums(built):
    for r in read_csv(s19e.GATE_COM):
        assert r["status_19e"] in s19e.STATUS_19E_COM_ALLOWED
    for r in read_csv(s19e.GATE_ART):
        assert r["status_artigo_slides"] in s19e.STATUS_ART_ALLOWED
