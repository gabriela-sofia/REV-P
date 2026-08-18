"""Tests for SUSC-17G extracao direta das features fisicas dos canarios."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_17g_extracao_direta_features_fisicas_canarios"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "preflight.json",
    OUT / "auditoria_insumos_extracao_fisica.csv",
    OUT / "matriz_features_fisicas_diretas_canarios.csv",
    OUT / "fila_insumos_externos_features_fisicas.csv",
    OUT / "reavaliacao_calibracao_17e_com_features_diretas.csv",
    OUT / "simulacao_sensibilidade_features_diretas_review_only.csv",
    OUT / "gate_extracao_direta_features_fisicas.csv",
    OUT / "resumo_por_canario.csv",
    OUT / "resumo_por_status.csv",
    OUT / "summary.json",
    REPORTS / "SUSC_17G_EXTRACAO_DIRETA_FEATURES_FISICAS_CANARIOS.md",
    SCHEMAS / "susc_17g_extracao_direta_features_fisicas_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17g_extracao_fisica_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17g_extracao_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT, text=True, capture_output=True, timeout=600, check=False,
    )


def base_matriz(**overrides) -> dict:
    row = {
        "canary_patch_id": "S17C6_CANARY_REC_00001",
        "candidate_event_id": "S17C_REF_0063",
        "geometry_id": "S17C5_GEOM_0063",
        "bbox": "-34.94,-8.00,-34.93,-7.99", "crs": "EPSG:4326",
        "elevation_mean": "64.6", "elevation_min": "40.0", "elevation_max": "80.0", "elevation_std": "8.0",
        "slope_mean": "8.24", "slope_min": "0.5", "slope_max": "20.0", "slope_std": "3.0",
        "HAND_mean": "25.15", "HAND_min": "0.0", "HAND_max": "40.0",
        "distance_to_water_min": "156.0", "distance_to_water_mean": "200.0",
        "TWI_mean": "7.75", "flow_accumulation_mean": "1.11",
        "drainage_context": "distancia_hidrica_min_m=156.0",
        "feature_source_mode": "direta_por_dem_e_hidrografia_local",
        "dem_source": "copernicus_dem_glo30_dominio_bacia_17c38",
        "drainage_source": "faixas_marginais_hidrografia_oficial_dados_recife_17c35",
        "features_diretas_completas": "true", "features_diretas_parciais": "false",
        "features_ausentes": "nenhuma",
        "qualidade_extracao": "media_metodo_reconstruido_resolucao_92m",
        "bloqueios_extracao": "nenhum",
        "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false",
        "not_ground_truth_reason": "feature fisica direta review-only",
    }
    row.update(overrides)
    return row


def base_simulacao(**overrides) -> dict:
    row = {
        "simulacao_id": "S17G_SIM_0001", "canary_patch_id": "S17C6_CANARY_REC_00001",
        "score_oficial": "false", "substituir_score_v6": "false", "usar_em_treino": "false",
        "ground_truth": "false", "score_v7_allowed": "false", "review_only": "true",
    }
    row.update(overrides)
    return row


# --- Build + validator + determinismo --------------------------------------
def test_build_validator_e_determinismo():
    result = run_script("build_susc_17g_extracao_direta_features_fisicas_canarios.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    cards_first = {path: path.read_bytes() for path in sorted((OUT / "cartoes_extracao").glob("*.md"))}
    assert cards_first
    result = run_script("build_susc_17g_extracao_direta_features_fisicas_canarios.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    assert {path: path.read_bytes() for path in sorted((OUT / "cartoes_extracao").glob("*.md"))} == cards_first
    result = run_script("validate_susc_17g_extracao_direta_features_fisicas_canarios.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_caminho_funcional_e_status():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["canarios_processados"] == 5
    assert summary["status_final_17g"] in {
        "17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL",
        "17G_FEATURES_FISICAS_DIRETAS_COMPLETAS",
        "17G_FEATURES_FISICAS_DIRETAS_PARCIAIS",
        "17G_EXTRATOR_OPERACIONAL_AGUARDANDO_INSUMO_EXTERNO",
        "17G_EXPLORATORIA_COM_FEATURES_DIRETAS",
    }
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False
    assert summary["ground_truth"] is False


# --- Secao 13: casos exigidos ----------------------------------------------
def test_dem_local_valido_extrai_elevation_slope():
    matriz = read_csv(OUT / "matriz_features_fisicas_diretas_canarios.csv")
    assert len(matriz) == 5
    for r in matriz:
        assert float(r["elevation_mean"]) > 0
        assert float(r["slope_mean"]) >= 0
        assert r["dem_source"] not in {"", "not_available"}


def test_hidrografia_local_valida_extrai_distance_to_water():
    matriz = read_csv(OUT / "matriz_features_fisicas_diretas_canarios.csv")
    for r in matriz:
        assert float(r["distance_to_water_min"]) >= 0
        assert r["drainage_source"] not in {"", "not_available"}


def test_ausencia_de_dem_gera_fila_de_insumo_externo():
    s = _load_common()
    synthetic = [{"canary_patch_id": "S17C6_CANARY_REC_99999", "bbox": "-34.9,-8.0,-34.8,-7.9", "extracted": {}}]
    rows = s.fila_rows(synthetic)
    assert rows
    assert any(r["prioridade"] == "alta" and r["canary_patch_id"] == "S17C6_CANARY_REC_99999" for r in rows)


def test_referencia_comparativa_nao_vira_feature_direta():
    s = _load_common()
    row = base_matriz(dem_source="recife_00552_comparativo")
    assert any("referencia_comparativa_como_direta" in e for e in s.validate_matriz_rows([row]))


def test_score_exploratorio_nao_e_oficial():
    s = _load_common()
    row = base_simulacao(score_oficial="true")
    assert any("score_oficial_proibido_true" in e for e in s.validate_simulacao_rows([row]))


def test_score_exploratorio_nao_substitui_score_v6():
    s = _load_common()
    row = base_simulacao(substituir_score_v6="true")
    assert any("substituir_score_v6_proibido_true" in e for e in s.validate_simulacao_rows([row]))


def test_ground_truth_treino_score_v7_proibidos():
    s = _load_common()
    row = base_matriz(ground_truth="true", eligible_for_training="true", score_v7_allowed="true")
    errors = s.validate_matriz_rows([row])
    assert any("ground_truth_true_proibido" in e for e in errors)
    assert any("eligible_for_training_true_proibido" in e for e in errors)
    assert any("score_v7_allowed_true_proibido" in e for e in errors)


def test_valor_fisico_sem_fonte_local_falha():
    s = _load_common()
    row = base_matriz(dem_source="not_available", drainage_source="not_available")
    assert any("valor_fisico_sem_fonte_local" in e for e in s.validate_matriz_rows([row]))


def test_feature_completa_com_ausente_falha():
    s = _load_common()
    row = base_matriz(features_diretas_completas="true", features_ausentes="HAND;TWI")
    assert any("completa_com_feature_ausente" in e for e in s.validate_matriz_rows([row]))


def test_vocabulario_publico_proibido_falha():
    s = _load_common()
    hits = s.public_text_violations_text("texto publico com agente, LLM e IA")
    assert {"agente", "LLM", "IA"}.issubset(set(hits))


def test_status_invalido_falha():
    s = _load_common()
    row = base_matriz(feature_source_mode="modo_inexistente")
    assert any("feature_source_mode_fora_enum" in e for e in s.validate_matriz_rows([row]))


def test_justificativa_vazia_falha():
    s = _load_common()
    reav = {"canary_patch_id": "X", "justificativa_tecnica": ""}
    assert any("reavaliacao_sem_justificativa" in e for e in s.validate_reavaliacao_rows([reav]))


def test_direct_topo_idx_e_distancia_pura():
    s = _load_common()
    # descritor topografico: HAND alto e elevacao alta -> baixa suscetibilidade topografica
    low = s._direct_topo_idx({"HAND_mean": 25.0, "elevation_mean": 50.0, "TWI_mean": 0.0})
    high = s._direct_topo_idx({"HAND_mean": 0.0, "elevation_mean": 0.0, "TWI_mean": 15.0})
    assert low is not None and high is not None
    assert high > low
    # distancia ponto-segmento em metros: ponto sobre o segmento -> 0
    d = s._point_segment_dist(0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    assert abs(d) < 1e-9
