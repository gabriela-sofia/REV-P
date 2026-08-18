"""Tests for SUSC-18B execucao de geometrias regionais e separacao de fenomeno."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_18b_execucao_geometrias_regionais_separacao_fenomeno"
CARDS = OUT / "cartoes_regionais"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "preflight.json",
    OUT / "curitiba_execucao_geometria.csv",
    OUT / "curitiba_geometrias_normalizadas.csv",
    OUT / "curitiba_tarefas_externas_geometria.csv",
    OUT / "petropolis_separacao_fenomeno_executada.csv",
    OUT / "petropolis_candidatos_inundacao_separada.csv",
    OUT / "petropolis_bloqueios_fenomeno.csv",
    OUT / "petropolis_tarefas_externas_separacao.csv",
    OUT / "footprints_tecnicos_18b.csv",
    OUT / "fila_sar_18b.csv",
    OUT / "vinculos_evento_patch_18b.csv",
    OUT / "features_regionais_18b.csv",
    OUT / "fila_features_regionais_18b.csv",
    OUT / "matriz_referencia_observacional_18b.csv",
    OUT / "comparacao_regional_recife_curitiba_petropolis.csv",
    OUT / "gate_prontidao_17b_pos_18b.csv",
    OUT / "resumo_por_regiao.csv",
    OUT / "resumo_por_status.csv",
    OUT / "resumo_por_fenomeno.csv",
    OUT / "summary.json",
    REPORTS / "SUSC_18B_EXECUCAO_GEOMETRIAS_REGIONAIS_SEPARACAO_FENOMENO.md",
    SCHEMAS / "susc_18b_geometrias_fenomeno_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_18b_geometrias_fenomeno_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s18b_geometrias_common", path)
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


# --- 1) Curitiba com geometria oficial real vira geometria resolvida -------
def test_curitiba_geometria_oficial_vira_resolvida():
    cand = {"has_official_geometry": "true", "has_point": "true", "event_date_candidate": "2022-01-15"}
    assert S.classify_geometria_execucao(cand) == "geometria_oficial_resolvida"
    assert "geometria_oficial_resolvida" in S.GEOM_EXEC_FORTE


# --- 2) Curitiba sem geometria gera tarefa externa -------------------------
def test_curitiba_sem_geometria_gera_tarefa_externa():
    cand = {"has_official_geometry": "false", "has_point": "false",
            "event_date_candidate": "2022-01-15..2022-01-16",
            "geometry_status": "documentary_or_administrative_only"}
    assert S.classify_geometria_execucao(cand) == "geometria_ausente_com_tarefa_externa"
    tarefas = _read(OUT / "curitiba_tarefas_externas_geometria.csv")
    assert len(tarefas) >= 1
    for t in tarefas:
        assert t["formato_esperado"] == "geojson_ou_shapefile"
        assert t["expected_output_path"].startswith("local_runs/")


# --- 3) bairro/endereco textual e bloqueado --------------------------------
def test_endereco_textual_bloqueado():
    cand = {"has_address": "true", "has_point": "false", "event_date_candidate": "2023-10-28",
            "geometry_status": "address_text_only"}
    assert S.classify_geometria_execucao(cand) == "geometria_textual_bloqueada"
    assert "geometria_textual_bloqueada" not in S.GEOM_EXEC_FORTE
    # nenhuma geometria normalizada com origem textual
    norm = _read(OUT / "curitiba_geometrias_normalizadas.csv")
    assert all(r["geometria_de_ocorrencia"] == "true" for r in norm)


# --- 4) Petropolis misto e bloqueado ---------------------------------------
def test_petropolis_misto_bloqueado():
    assert S._phenomenon_class("hydrometeorological_or_unknown") == "fenomeno_misto"
    bloq = _read(OUT / "petropolis_bloqueios_fenomeno.csv")
    mistos = [r for r in bloq if r["classe_fenomeno"] == "fenomeno_misto"]
    assert mistos
    sep = _read(OUT / "petropolis_separacao_fenomeno_executada.csv")
    for r in sep:
        if r["classe_fenomeno"] in {"fenomeno_misto", "deslizamento"}:
            assert r["pode_seguir_como_inundacao"] == "false"


# --- 5) Petropolis com inundacao separada pode seguir ----------------------
def test_petropolis_inundacao_separada_pode_seguir():
    assert S._phenomenon_class("flood_inundation_alagamento") == "inundacao_alagamento_enxurrada"
    inund = _read(OUT / "petropolis_candidatos_inundacao_separada.csv")
    assert len(inund) >= 1
    for r in inund:
        assert r["classe_fenomeno"] == "inundacao_alagamento_enxurrada"
        assert r["status_referencia_observacional"] == "evidencia_contextual"


# --- 6) footprint SAR sem raster local gera fila ---------------------------
def test_footprint_sar_sem_raster_gera_fila():
    fila = _read(OUT / "fila_sar_18b.csv")
    assert len(fila) >= 1
    for r in fila:
        assert r["colecao_esperada"] == "sentinel-1-rtc"
        assert r["command_hint"].strip()
        assert r["depende_de"].strip()
    fps = _read(OUT / "footprints_tecnicos_18b.csv")
    assert any(r["footprint_status"] == "fila_execucao_externa" for r in fps)


# --- 7) geometria resolvida gera patch-link forte (Recife herdado) ---------
def test_geometria_resolvida_gera_patch_link():
    vinc = _read(OUT / "vinculos_evento_patch_18b.csv")
    fortes = [r for r in vinc if r["vinculo_forte"] == "true"]
    assert len(fortes) == 5
    for r in fortes:
        assert r["classe_vinculo"] == "exact_polygon_overlap"
        assert r["geometry_id"] not in {"", "not_available"}
        assert r["patch_id"] not in {"", "not_available"}


# --- 8) same_region_only nao e forte ---------------------------------------
def test_same_region_only_nao_e_forte():
    vinc = _read(OUT / "vinculos_evento_patch_18b.csv")
    regionais = [r for r in vinc if r["classe_vinculo"] == "same_region_only"]
    assert regionais
    assert all(r["vinculo_forte"] == "false" for r in regionais)
    assert "same_region_only" not in S.CLASSE_VINCULO_FORTE


# --- 9) feature sem fonte falha (validator) --------------------------------
def test_validator_detecta_feature_sem_fonte():
    rows = [{"vinculo_id": "F", "fisico_elevacao_media": "40.0", "fonte_fisico": "",
             "espectral_disponivel": "false", "chuva_disponivel": "false",
             "feature_pre_evento_apenas": "true"}]
    errs = S.validate_features_rows(rows)
    assert any("feature_fisica_sem_fonte" in e for e in errs)


# --- 10) ground_truth/trainable/score_v7 proibidos -------------------------
def test_hard_defaults_proibidos():
    matriz = _read(OUT / "matriz_referencia_observacional_18b.csv")
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


def test_validator_detecta_score_v7_true():
    rows = [{"item_id": "X", "ground_truth": "false", "eligible_for_training": "false",
             "score_v7_allowed": "true", "review_only": "true", "classe_vinculo": "same_region_only",
             "status_referencia_observacional": "referencia_observacional_parcial",
             "uso_permitido": "avaliacao_somente_revisao", "justificativa_tecnica": "x"}]
    errs = S.validate_matriz_rows(rows)
    assert any("score_v7_allowed_true_proibido" in e for e in errs)


def test_validator_detecta_forte_sem_geometry_id():
    rows = [{"item_id": "Y", "ground_truth": "false", "eligible_for_training": "false",
             "score_v7_allowed": "false", "review_only": "true", "classe_vinculo": "exact_polygon_overlap",
             "geometry_id": "not_available", "patch_id": "P1",
             "status_referencia_observacional": "referencia_observacional_forte_somente_revisao",
             "uso_permitido": "avaliacao_somente_revisao", "justificativa_tecnica": "x"}]
    errs = S.validate_matriz_rows(rows)
    assert any("vinculo_forte_sem_geometry_id" in e for e in errs)


# --- 11) 17B nao e criado --------------------------------------------------
def test_17b_nao_criado():
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summ["benchmark_17b_criado"] is False
    assert summ["status_17b"] in S.STATUS_17B_ALLOWED
    assert summ["status_final_18b"] in S.GATE_FINAL_ALLOWED
    assert not any(p.name.startswith("benchmark_17b") for p in OUT.glob("*.csv"))


# --- 12) vocabulario publico proibido falha --------------------------------
def test_vocabulario_publico_proibido():
    for termo in ["agente", "agentic", "IA", "LLM", "Codex"]:
        assert S.public_text_violations_text(f"texto com {termo} aqui"), termo
    assert S.public_text_violations_files() == []


def test_vocabulario_nao_gera_falso_positivo():
    limpo = "referencia observacional regional com avaliacao somente revisao e separacao de fenomeno"
    assert S.public_text_violations_text(limpo) == []


# --- Curitiba: patches candidatos region-only nomeados (avanco concreto) ---
def test_curitiba_patches_region_only_identificados():
    exec_rows = _read(OUT / "curitiba_execucao_geometria.csv")
    alvo = [r for r in exec_rows if r["candidate_event_id"] == "S17C_REF_0060"]
    assert alvo, "evento datado de Curitiba de 2022 esperado"
    assert int(alvo[0]["patches_candidatos_region_only"]) > 0
    assert alvo[0]["status_execucao_geometria"] == "geometria_ausente_com_tarefa_externa"


# --- Recife forte carrega features reais -----------------------------------
def test_recife_forte_com_features_reais():
    feats = _read(OUT / "features_regionais_18b.csv")
    assert len(feats) == 5
    for r in feats:
        assert float(r["fisico_elevacao_media"]) > 0
        assert r["fonte_fisico"].strip()
        assert r["feature_pre_evento_apenas"] == "true"
        assert r["chuva_disponivel"] == "true"
        assert r["espectral_disponivel"] == "true"


# --- geometria normalizada exige geometria de ocorrencia (validator) -------
def test_validator_geometria_normalizada_exige_ocorrencia():
    rows = [{"geometry_id": "G", "geometria_de_ocorrencia": "false"}]
    errs = S.validate_cur_norm_rows(rows)
    assert any("geometria_normalizada_nao_de_ocorrencia" in e for e in errs)
