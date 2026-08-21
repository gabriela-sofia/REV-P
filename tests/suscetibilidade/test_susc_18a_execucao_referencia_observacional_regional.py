"""Tests for SUSC-18A execucao de referencia observacional regional."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18a_execucao_referencia_observacional_regional"
CARDS = OUT / "cartoes_regionais"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "preflight.json",
    OUT / "execucao_fila_regional_17i.csv",
    OUT / "curitiba_geometrias_observacionais.csv",
    OUT / "curitiba_fila_obtencao_geometria.csv",
    OUT / "petropolis_classificacao_fenomeno.csv",
    OUT / "petropolis_fila_separacao_fenomeno.csv",
    OUT / "footprints_tecnicos_regionais.csv",
    OUT / "fila_execucao_footprint_sar.csv",
    OUT / "vinculos_regionais_evento_patch.csv",
    OUT / "features_regionais_por_vinculo.csv",
    OUT / "fila_extracao_features_regionais.csv",
    OUT / "matriz_referencia_observacional_regional.csv",
    OUT / "gate_prontidao_17b_pos_18a.csv",
    OUT / "resumo_por_regiao.csv",
    OUT / "resumo_por_status.csv",
    OUT / "resumo_por_fenomeno.csv",
    OUT / "summary.json",
    REPORTS / "SUSC_18A_EXECUCAO_REFERENCIA_OBSERVACIONAL_REGIONAL.md",
    SCHEMAS / "susc_18a_referencia_observacional_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_18a_referencia_observacional_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18a_referencia_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_common()


def _read(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def setup_module(module):
    S.build_all()


# --- Build/estrutura -------------------------------------------------------
def test_build_gera_todos_os_outputs():
    S.build_all()
    missing = [str(p) for p in EXPECTED if not p.exists()]
    assert not missing, f"faltando: {missing}"


def test_validator_passa():
    assert S.validate() == 0


# --- 1) Curitiba com ponto oficial vira geometria observacional ------------
def test_curitiba_com_ponto_oficial_vira_geometria():
    cand = {"has_official_geometry": "true", "has_point": "true", "has_bbox": "false"}
    assert S.classify_geometria(cand) == "geometria_oficial_ponto"
    assert "geometria_oficial_ponto" in S.CLASSE_GEOMETRIA_FORTE


# --- 2) Curitiba sem geometria gera fila -----------------------------------
def test_curitiba_sem_geometria_gera_fila():
    cand = {"has_official_geometry": "false", "has_point": "false",
            "geometry_status": "documentary_or_administrative_only"}
    assert S.classify_geometria(cand) == "sem_geometria"
    fila = _read(OUT / "curitiba_fila_obtencao_geometria.csv")
    assert len(fila) >= 1
    # todos os eventos de inundacao datados de Curitiba entram na fila
    geoms = _read(OUT / "curitiba_geometrias_observacionais.csv")
    assert all(r["classe_geometria"] in {"sem_geometria", "endereco_textual_sem_geometria"} for r in geoms)


def test_curitiba_endereco_textual_nao_e_geometria_forte():
    cand = {"has_address": "true", "has_point": "false", "geometry_status": "address_text_only"}
    assert S.classify_geometria(cand) == "endereco_textual_sem_geometria"
    assert S.classify_geometria(cand) not in S.CLASSE_GEOMETRIA_FORTE


# --- 3) Petropolis misto e bloqueado ---------------------------------------
def test_petropolis_misto_bloqueado():
    assert S._phenomenon_class("hydrometeorological_or_unknown") == "fenomeno_misto"
    cand = {"phenomenon_type": "hydrometeorological_or_unknown", "event_date_candidate": "2022-02-20",
            "authority_tier": "official", "source_type": "official_observed_event_point"}
    assert S._classify_regional(cand) == "bloqueado_por_fenomeno"
    pet = _read(OUT / "petropolis_classificacao_fenomeno.csv")
    mistos = [r for r in pet if r["classe_fenomeno"] == "fenomeno_misto"]
    assert mistos, "esperado ao menos um fenomeno misto em Petropolis"
    assert all(r["pode_seguir_como_inundacao"] == "false" for r in mistos)


# --- 4) Petropolis inundacao separada pode seguir --------------------------
def test_petropolis_inundacao_pode_seguir():
    assert S._phenomenon_class("flood_inundation_alagamento") == "inundacao_alagamento_enxurrada"
    pet = _read(OUT / "petropolis_classificacao_fenomeno.csv")
    inund = [r for r in pet if r["classe_fenomeno"] == "inundacao_alagamento_enxurrada"]
    assert inund, "esperado evento de inundacao em Petropolis"
    assert all(r["pode_seguir_como_inundacao"] == "true" for r in inund)


# --- 5) SAR sem raster local gera fila executavel --------------------------
def test_sar_sem_raster_gera_fila():
    fila = _read(OUT / "fila_execucao_footprint_sar.csv")
    assert len(fila) >= 1
    for r in fila:
        assert r["colecao_esperada"] == "sentinel-1-rtc"
        assert r["command_hint"].strip()
        assert r["expected_output_path"].startswith("local_runs/")
    fps = _read(OUT / "footprints_tecnicos_regionais.csv")
    assert any(r["possui_raster_local"] == "false" for r in fps)


# --- 6) geometry/footprint resolvido gera patch-link -----------------------
def test_geometria_footprint_resolvido_gera_patch_link():
    vinc = _read(OUT / "vinculos_regionais_evento_patch.csv")
    fortes = [r for r in vinc if r["vinculo_forte"] == "true"]
    assert len(fortes) == 5, "5 canarios de Recife com vinculo forte"
    for r in fortes:
        assert r["classe_vinculo"] == "exact_polygon_overlap"
        assert r["geometry_id"] not in {"", "not_available"}
        assert r["patch_id"] not in {"", "not_available"}


# --- 7) same_region_only nao e forte ---------------------------------------
def test_same_region_only_nao_e_forte():
    vinc = _read(OUT / "vinculos_regionais_evento_patch.csv")
    regionais = [r for r in vinc if r["classe_vinculo"] == "same_region_only"]
    assert regionais, "esperado vinculos regionais nao fortes"
    assert all(r["vinculo_forte"] == "false" for r in regionais)
    assert "same_region_only" not in S.CLASSE_VINCULO_FORTE


# --- 8) ground_truth/trainable/score_v7 proibidos --------------------------
def test_hard_defaults_proibidos():
    matriz = _read(OUT / "matriz_referencia_observacional_regional.csv")
    assert matriz
    for r in matriz:
        assert r["ground_truth"] == "false"
        assert r["eligible_for_training"] == "false"
        assert r["score_v7_allowed"] == "false"
        assert r["review_only"] == "true"
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summ["ground_truth_true_count"] == 0
    assert summ["eligible_for_training_true_count"] == 0
    assert summ["score_v7_allowed_true_count"] == 0
    assert summ["score_v6_changed"] is False
    assert summ["score_v7_created"] is False


def test_validator_detecta_ground_truth_true():
    rows = [{"item_id": "X", "ground_truth": "true", "eligible_for_training": "false",
             "score_v7_allowed": "false", "review_only": "true", "classe_vinculo": "same_region_only",
             "status_referencia_observacional": "referencia_observacional_parcial",
             "uso_permitido": "avaliacao_somente_revisao", "justificativa_tecnica": "x"}]
    errs = S.validate_matriz_rows(rows)
    assert any("ground_truth_true_proibido" in e for e in errs)


# --- 9) 17B nao e criado (nenhum benchmark) --------------------------------
def test_17b_nao_criado():
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summ["benchmark_17b_criado"] is False
    assert summ["status_17b"] in S.STATUS_17B_ALLOWED
    assert summ["status_final_18a"] in S.GATE_FINAL_ALLOWED
    # nenhum arquivo de benchmark 17B foi criado nesta pasta
    assert not any(p.name.startswith("benchmark_17b") for p in OUT.glob("*.csv"))


# --- 10) vocabulario publico proibido falha --------------------------------
def test_vocabulario_publico_proibido():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo} aqui"), termo
    # os outputs publicos reais nao contem vocabulario proibido
    assert S.public_text_violations_files() == []


def test_vocabulario_nao_gera_falso_positivo():
    # palavras tecnicas legitimas nao disparam o filtro
    limpo = "referencia observacional regional com avaliacao somente revisao e criterio de prontidao"
    assert S.public_text_violations_text(limpo) == []


# --- Fenomeno misto nunca vira inundacao (validator) -----------------------
def test_validator_detecta_misto_como_inundacao():
    rows = [{"candidate_event_id": "X", "classe_fenomeno": "fenomeno_misto",
             "pode_seguir_como_inundacao": "true"}]
    errs = S.validate_petropolis_rows(rows)
    assert any("fenomeno_misto_virou_inundacao" in e for e in errs)


# --- patch-link forte sem geometry_id falha (validator) --------------------
def test_validator_detecta_forte_sem_geometry_id():
    rows = [{"vinculo_id": "L", "classe_vinculo": "exact_polygon_overlap", "vinculo_forte": "true",
             "geometry_id": "not_available", "patch_id": "P1", "justificativa_tecnica": "x"}]
    errs = S.validate_vinculos_rows(rows)
    assert any("forte_sem_geometry_id" in e for e in errs)


# --- feature preenchida sem fonte falha (validator) ------------------------
def test_validator_detecta_feature_sem_fonte():
    rows = [{"vinculo_id": "F", "fisico_elevacao_media": "40.0", "fonte_fisico": "",
             "espectral_disponivel": "false", "chuva_disponivel": "false",
             "feature_pre_evento_apenas": "true"}]
    errs = S.validate_features_rows(rows)
    assert any("feature_fisica_sem_fonte" in e for e in errs)


# --- referencia forte de Recife carrega features reais ---------------------
def test_recife_forte_com_features_reais():
    feats = _read(OUT / "features_regionais_por_vinculo.csv")
    assert len(feats) == 5
    for r in feats:
        assert float(r["fisico_elevacao_media"]) > 0
        assert r["fonte_fisico"].strip()
        assert r["feature_pre_evento_apenas"] == "true"
        assert r["chuva_disponivel"] == "true"
        assert r["espectral_disponivel"] == "true"
