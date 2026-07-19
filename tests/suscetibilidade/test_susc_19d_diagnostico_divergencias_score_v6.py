"""Testes do SUSC-19D diagnostico de divergencias review-only do score_v6.

Cobrem: 7 patches observacionais herdados; curitiba_01101 como divergencia;
background nunca negativo; score_v6 intacto; score_v7 bloqueado; 17B nao criado;
hipoteses nao viram score; sensibilidade nao substitui score_v6; missingness
territorial explicito; output 19D versionavel; vocabulario publico proibido falha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "suscetibilidade"
sys.path.insert(0, str(SCRIPTS))

import susc_19d_diagnostico_score_v6_common as s19d  # noqa: E402
from susc_io import read_csv, read_json  # noqa: E402


@pytest.fixture(scope="module")
def built():
    summary = s19d.build_all()
    errors = s19d.validate_outputs(summary)
    return summary, errors


def test_validator_sem_erros(built):
    _summary, errors = built
    assert errors == [], f"validator reportou erros: {errors}"


def test_sete_patches_observacionais(built):
    diag = read_csv(s19d.DIAG_PATCHES)
    assert len(diag) == 7
    assert sum(1 for r in diag if r["regiao"] == "recife") == 5
    assert sum(1 for r in diag if r["regiao"] == "curitiba") == 2
    assert not any(r["regiao"] == "petropolis" for r in diag)


def test_observados_herdados_do_19c(built):
    diag_ids = {r["patch_id"] for r in read_csv(s19d.DIAG_PATCHES)}
    herdados = {r["patch_id"] for r in read_csv(s19d.REG_OBS_19C)}
    assert diag_ids == herdados


def test_curitiba_01101_e_divergencia_forte(built):
    diag = {r["patch_id"]: r for r in read_csv(s19d.DIAG_PATCHES)}
    assert "curitiba_01101" in diag
    r = diag["curitiba_01101"]
    assert r["score_v6_class"].lower() == "low"
    blob = " ".join(str(v).lower() for v in r.values())
    assert "diverg" in blob
    summary, _ = built
    assert summary["divergencia_forte"] == "curitiba_01101"


def test_background_nunca_e_negativo(built):
    for path in s19d._public_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore").lower()
        for m in s19d.NEGATIVE_RE.finditer(txt):
            window = txt[max(0, m.start() - 40): m.end() + 40]
            assert (
                "nao e negativo" in window
                or "nunca e negativo" in window
                or "nunca negativo" in window
            ), f"termo negativo sem ressalva em {path.name}"


def test_score_v6_intacto(built):
    summary, _ = built
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False


def test_score_v7_bloqueado(built):
    summary, _ = built
    assert summary["status_score_v7"] in s19d.STATUS_V7_ALLOWED
    assert summary["status_score_v7"] != "SCORE_V7_AUTORIZADO"
    assert summary["score_v7_allowed_true_count"] == 0
    for r in read_csv(s19d.GATE_V7):
        assert r["status_score_v7"] in s19d.STATUS_V7_ALLOWED


def test_17b_nao_criado(built):
    summary, _ = built
    assert summary["marco_17b_criado"] is False
    assert summary["benchmark_17b_criado"] is False
    assert summary["status_17b"] == "17B_NAO_CRIADO"


def test_hipoteses_nao_viram_score(built):
    hip = read_csv(s19d.HIPOTESES)
    assert len(hip) == 7
    for r in hip:
        assert r["pode_virar_score_v7_agora"] == "false"


def test_sensibilidade_nao_substitui_score_v6(built):
    sens = read_csv(s19d.SENSIBILIDADE)
    assert sens
    for r in sens:
        assert r["substitui_score_v6"] == "false"
        assert r["score_oficial"] == "false"
        assert r["score_v7"] == "false"
        assert r["uso_permitido"] == "analise_sensibilidade_review_only"
    # ha cenarios executados e cenarios de desenho (nao executados)
    assert any(r["executado"] == "true" for r in sens)
    assert any(r["executado"] == "false" for r in sens)


def test_missingness_territorial_aparece(built):
    dec_blob = " ".join(
        " ".join(str(v) for v in r.values()) for r in read_csv(s19d.DECOMP_FAMILIAS)
    )
    assert "MapBiomas" in dec_blob
    assert "impervious_proxy" in dec_blob
    tip_types = {r["failure_type"] for r in read_csv(s19d.TIPOLOGIA)}
    assert "missingness_territorial" in tip_types


def test_inventario_documental_rainfall_urban(built):
    inv = read_csv(s19d.INVENTARIO)
    comps = {r["componente"] for r in inv}
    assert "evidence_support_index" in comps
    assert "rainfall_trigger_index" in comps
    assert "urban_spectral_index" in comps
    assert any(r["entra_documental"] == "true" for r in inv)
    assert any(r["entra_rainfall"] == "true" for r in inv)
    # pesos explicitos existem -> componentes formais nao ausentes
    summary, _ = built
    assert summary["componentes_formais_ausentes"] is False


def test_decomposicao_aderencia_e_divergencia(built):
    summary, _ = built
    # aderencia: fisica/urbana/espectral; divergencia: hidrologica/chuva
    assert "fisica_topografica" in summary["familias_aderentes"]
    assert "hidrologica" in summary["familias_divergentes"]
    assert "chuva_hidrometeorologica" in summary["familias_divergentes"]


def test_tipologia_enums_e_bloqueio_v7(built):
    tip = read_csv(s19d.TIPOLOGIA)
    for r in tip:
        assert r["failure_type"] in s19d.FAILURE_TYPES_ALLOWED
        assert r["bloqueia_score_v7"] == "true"
    types = {r["failure_type"] for r in tip}
    assert {"amostra_minima", "hidrologia_divergente"} <= types


def test_output_19d_versionavel(built):
    ignored = s19d.git_ignored([s19d.SUMMARY, s19d.DIAG_PATCHES])
    assert s19d.rel(s19d.SUMMARY) not in ignored
    assert s19d.rel(s19d.DIAG_PATCHES) not in ignored


def test_vocabulario_publico_proibido_falha():
    for termo in ("agente", "agentic", "codex", "LLM", "IA"):
        assert s19d.public_text_violations_text(f"texto com {termo} embutido")


def test_vocabulario_publico_limpo_nos_outputs(built):
    assert s19d.validate_public_text() == []


def test_conclusao_estatistica_forte_bloqueada():
    # o validator deve barrar p-valor / significancia estatistica forte
    assert s19d.STRONG_STATS_RE.search("resultado com p-valor 0.01")
    assert s19d.STRONG_STATS_RE.search("estatisticamente significativo")


def test_schema_gerado(built):
    schema = read_json(s19d.SCHEMA)
    assert schema["guardrails"]["score_v6_changed"]["const"] == "false"
    assert schema["guardrails"]["score_v7"]["const"] == "false"
    assert "SCORE_V7_APENAS_HIPOTESES_REVIEW_ONLY" in schema["status_score_v7_allowed"]


def test_patch_stats_sar_nao_e_feature(built):
    for path in (s19d.DECOMP_FAMILIAS, s19d.DIAG_PATCHES):
        for r in read_csv(path):
            blob = " ".join(str(v).lower() for v in r.values())
            assert "patch_stats" not in blob
