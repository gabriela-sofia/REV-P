"""Testes do SUSC-19F execucao MapBiomas/GEE e ingestao territorial.

Cobrem: ambiente sem auth gera bloqueio especifico; CSV real preenche matriz;
ausencia de CSV nao inventa feature; patch_id unico; 300 patches preservados;
score_v6 intacto; score_v7 bloqueado; 17B nao criado; coverage_score nao e
suscetibilidade; output 19F versionavel; vocabulario publico proibido falha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "suscetibilidade"
sys.path.insert(0, str(SCRIPTS))

import susc_19f_mapbiomas_common as s19f  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402


@pytest.fixture(scope="module")
def built():
    summary = s19f.build_all()
    errors = s19f.validate_outputs(summary)
    return summary, errors


def test_validator_sem_erros(built):
    _summary, errors = built
    assert errors == [], f"validator reportou erros: {errors}"


def test_ambiente_sem_auth_bloqueia_especifico():
    # funcao pura de status: sem auth e sem CSV -> bloqueio por autenticacao
    assert s19f.status_19f("gee_nao_autenticado", False, 0) == "19F_BLOQUEADO_POR_AUTENTICACAO"
    assert s19f.status_19f("permissao_insuficiente", False, 0) == "19F_BLOQUEADO_POR_PERMISSAO"
    assert s19f.status_19f("biblioteca_ausente", False, 0) == "19F_BLOQUEADO_FAIL_CLOSED"


def test_ausencia_de_csv_nao_inventa_feature():
    manifest = s19f.load_manifest()
    rows, meta = s19f.matriz_territorial_rows(manifest, {})  # sem CSV real
    assert meta["filled"] == 0
    for r in rows:
        assert r["fonte_real"] == "false"
        assert r["territorial_status"] == "aguardando_export"
        for f in ("water_prop", "exposed_soil_prop", "impervious_proxy", "mapbiomas_class_majority"):
            assert r[f] == s19f.NA


def test_csv_real_preenche_matriz(built):
    summary, _ = built
    # o CSV local foi extraido nesta sessao (300 patches reais)
    if summary["csv_recuperado"]:
        terr = read_csv(s19f.MAT_TERRITORIAL)
        filled = [r for r in terr if r["fonte_real"] == "true"]
        assert len(filled) == summary["patches_preenchidos"]
        for r in filled:
            assert s19f._num(r["water_prop"]) is not None
            assert r["landcover_source"] == s19f.LANDCOVER_SOURCE
            assert r["landcover_reference_year"] == str(s19f.MAPBIOMAS_YEAR)
    else:
        pytest.skip("CSV real ausente nesta execucao")


def test_patch_id_unico(built):
    for path in (s19f.MAT_TERRITORIAL, s19f.MAT_MULTIMODAL):
        ids = [r["patch_id"] for r in read_csv(path)]
        assert len(ids) == len(set(ids)), f"patch_id duplicado em {path.name}"


def test_300_patches_preservados(built):
    mm = read_csv(s19f.MAT_MULTIMODAL)
    assert len(mm) == s19f.EXPECTED_PATCHES
    terr = read_csv(s19f.MAT_TERRITORIAL)
    assert len(terr) == s19f.EXPECTED_PATCHES


def test_score_v6_intacto(built):
    summary, _ = built
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False


def test_score_v7_bloqueado(built):
    summary, _ = built
    assert summary["status_score_v7"] in s19f.STATUS_V7_ALLOWED
    assert summary["score_v7_allowed_true_count"] == 0
    for r in read_csv(s19f.GATE_V7):
        assert r["status_score_v7"] in s19f.STATUS_V7_ALLOWED


def test_17b_nao_criado(built):
    summary, _ = built
    assert summary["status_17b"] == "17B_NAO_CRIADO"
    assert summary["marco_17b_criado"] is False
    assert summary["benchmark_17b_criado"] is False


def test_coverage_nao_e_suscetibilidade(built):
    summary, _ = built
    assert summary["coverage_is_susceptibility"] is False
    for r in read_csv(s19f.MAT_MULTIMODAL):
        assert r["coverage_score_is_susceptibility"] == "false"
    for r in read_csv(s19f.AUD_COBERTURA):
        assert r["coverage_is_susceptibility"] == "false"


def test_hard_defaults_matriz(built):
    for r in read_csv(s19f.MAT_MULTIMODAL):
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed",
                      "score_oficial", "substituir_score_v6", "usar_em_treino"):
            assert r[field] == "false"


def test_sar_nao_e_feature_territorial(built):
    for path in (s19f.MAT_TERRITORIAL, s19f.MAT_MULTIMODAL):
        for r in read_csv(path):
            for k, v in r.items():
                kl = k.lower()
                if any(t in kl for t in ("water_prop", "exposed_soil", "impervious", "mapbiomas")):
                    assert "sar" not in str(v).lower()


def test_proportions_from_hist_pura():
    hist = {"24": 646.0, "3": 253.0, "33": 10.0, "23": 5.0}
    d = s19f.proportions_from_hist(hist)
    assert d["mapbiomas_class_majority"] == 24
    total = 646 + 253 + 10 + 5
    assert abs(d["water_prop"] - 10 / total) < 1e-6       # classe 33
    assert abs(d["impervious_proxy"] - 646 / total) < 1e-6  # classe 24
    assert abs(d["exposed_soil_prop"] - 5 / total) < 1e-6   # classe 23
    # histograma vazio nao inventa
    assert s19f.proportions_from_hist({}) == {}


def test_cobertura_territorial_subiu(built):
    summary, _ = built
    if summary["patches_preenchidos"] == s19f.EXPECTED_PATCHES:
        assert summary["cobertura_territorial_19f"] > summary["cobertura_territorial_19a"]
        assert abs(summary["cobertura_territorial_19f"] - 1.0) < 1e-6


def test_output_19f_versionavel(built):
    ignored = s19f.git_ignored([s19f.SUMMARY, s19f.MAT_TERRITORIAL])
    assert s19f.rel(s19f.SUMMARY) not in ignored
    assert s19f.rel(s19f.MAT_TERRITORIAL) not in ignored


def test_sem_credencial_em_output_publico(built):
    for path in s19f._public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        assert not s19f.SECRET_RE.search(txt), f"credencial em {path.name}"


def test_vocabulario_publico_proibido_falha():
    for termo in ("agente", "agentic", "codex", "LLM", "IA"):
        assert s19f.public_text_violations_text(f"texto com {termo} embutido")


def test_vocabulario_publico_limpo(built):
    assert s19f.validate_public_text() == []


def test_schema_gerado(built):
    schema = read_json(s19f.SCHEMA)
    assert schema["guardrails"]["score_v6_changed"]["const"] == "false"
    assert schema["coverage_is_susceptibility"] is False
    assert "SCORE_V7_NAO_AUTORIZADO" in schema["status_score_v7_allowed"]
