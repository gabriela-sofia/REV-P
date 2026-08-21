"""Tests for SUSC-17F extracao fisica e topografica dos canarios observacionais."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17f_extracao_fisica_topografica_canarios_observacionais"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "preflight_extracao_fisica.csv",
    OUT / "preflight_extracao_fisica.json",
    OUT / "inventario_fontes_fisicas_topograficas.csv",
    OUT / "matriz_fisica_topografica_canarios.csv",
    OUT / "reavaliacao_prontidao_17e_com_features_fisicas.csv",
    OUT / "matriz_calibracao_exploratoria_aprimorada.csv",
    OUT / "simulacao_sensibilidade_componente_fisico_review_only.csv",
    OUT / "hipoteses_ajuste_pesos_componente_fisico.csv",
    OUT / "fila_extracao_features_fisicas_pendentes.csv",
    OUT / "gate_features_fisicas_canarios.csv",
    OUT / "resumo_por_status.csv",
    OUT / "resumo_por_canario.csv",
    OUT / "summary.json",
    REPORTS / "SUSC_17F_EXTRACAO_FISICA_TOPOGRAFICA_CANARIOS_OBSERVACIONAIS.md",
    SCHEMAS / "susc_17f_extracao_fisica_topografica_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17f_extracao_fisica_topografica_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17f_extracao_common", path)
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
        "region": "REC",
        "city": "Recife",
        "geometry_id": "S17C5_GEOM_0063",
        "feature_source_mode": "referencia_comparativa_review_only",
        "HAND": "5.2599", "elevation": "8.0504", "slope": "2.1454",
        "distance_to_water": "1490.9467", "TWI": "123.5202", "flow_accumulation": "1.6799",
        "drainage_context": "distancia_hidrica_referencia_m=1490.9467",
        "features_fisicas_completas": "false",
        "features_fisicas_parciais": "false",
        "features_fisicas_ausentes": "HAND;elevation;slope;distance_to_water;TWI;flow_accumulation",
        "referencia_patch_id": "recife_00552",
        "referencia_distance_m": "984.4000",
        "referencia_overlap_ratio": "0.0000",
        "uso_permitido": "referencia_comparativa_review_only",
        "ground_truth": "false",
        "eligible_for_training": "false",
        "score_v7_allowed": "false",
        "not_ground_truth_reason": "referencia comparativa review-only",
    }
    row.update(overrides)
    return row


def base_calibracao(**overrides) -> dict:
    row = {
        "canary_patch_id": "S17C6_CANARY_REC_00001",
        "justificativa_tecnica": "componente fisico comparativo penalizado review-only",
        "score_oficial": "false",
        "substituir_score_v6": "false",
        "usar_em_treino": "false",
        "ground_truth": "false",
        "review_only": "true",
    }
    row.update(overrides)
    return row


# --- Build + validator + determinismo --------------------------------------
def test_build_validator_e_determinismo():
    result = run_script("build_susc_17f_extracao_fisica_topografica_canarios_observacionais.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    cards_first = {path: path.read_bytes() for path in sorted((OUT / "cartoes_fisicos").glob("*.md"))}
    assert cards_first
    result = run_script("build_susc_17f_extracao_fisica_topografica_canarios_observacionais.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    assert {path: path.read_bytes() for path in sorted((OUT / "cartoes_fisicos").glob("*.md"))} == cards_first
    result = run_script("validate_susc_17f_extracao_fisica_topografica_canarios_observacionais.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_caminho_funcional_e_status():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["canarios_avaliados"] == 5
    assert summary["referencia_comparativa"] == 5
    assert summary["fila_extracao"] == 5
    assert summary["calibracao_exploratoria_aprimorada"] == 5
    assert summary["features_fisicas_diretas"] == 0
    assert summary["pode_calibracao_forte_agora"] == 0
    assert summary["status_final_17f"] == "17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO"
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False


def test_todos_canarios_recebem_status():
    matriz = read_csv(OUT / "matriz_fisica_topografica_canarios.csv")
    modes = {"feature_direta_review_only", "referencia_comparativa_review_only", "fila_extracao_necessaria"}
    assert len(matriz) == 5
    assert all(r["feature_source_mode"] in modes for r in matriz)


# --- Secao 12: casos exigidos ----------------------------------------------
def test_feature_direta_permite_consolidacao_fisica():
    s = _load_common()
    items = [{"direct_available": True, "feature_source_mode": "feature_direta_review_only",
              "pode_forte_agora": True} for _ in range(5)]
    assert s._final_status(items, [{"canary_patch_id": "x"}], []) == "17F_FEATURES_FISICAS_DIRETAS_CONSOLIDADAS"
    # matriz com feature direta e sem referencia nao gera erro de comparativa-como-direta
    row = base_matriz(feature_source_mode="feature_direta_review_only", referencia_patch_id="not_available",
                      referencia_distance_m="not_available")
    assert not any("comparativa_apresentada_como_direta" in e for e in s.validate_matriz_rows([row]))


def test_referencia_comparativa_nao_vira_feature_direta():
    s = _load_common()
    row = base_matriz(feature_source_mode="feature_direta_review_only", referencia_patch_id="recife_00552")
    assert any("comparativa_apresentada_como_direta" in e for e in s.validate_matriz_rows([row]))


def test_ausencia_de_feature_gera_fila_executavel():
    fila = read_csv(OUT / "fila_extracao_features_fisicas_pendentes.csv")
    canaries = {r["canary_patch_id"] for r in fila}
    assert len(canaries) == 5
    # cada tarefa e concreta: bbox, crs e fonte
    for r in fila:
        assert r["bbox"] not in {"", "not_available"}
        assert r["crs"] == "EPSG:4326"
        assert r["data_source_needed"]
        assert r["command_hint"]


def test_score_exploratorio_nao_e_oficial():
    s = _load_common()
    row = base_calibracao(score_oficial="true")
    assert any("score_exploratorio_marcado_oficial" in e for e in s.validate_calibracao_rows([row]))


def test_score_exploratorio_nao_substitui_score_v6():
    s = _load_common()
    row = base_calibracao(substituir_score_v6="true")
    assert any("substitui_score_v6" in e for e in s.validate_calibracao_rows([row]))


def test_ground_truth_treino_score_v7_proibidos():
    s = _load_common()
    row = base_matriz(ground_truth="true", eligible_for_training="true", score_v7_allowed="true")
    errors = s.validate_matriz_rows([row])
    assert any("ground_truth_true_proibido" in e for e in errors)
    assert any("eligible_for_training_true_proibido" in e for e in errors)
    assert any("score_v7_allowed_true_proibido" in e for e in errors)


def test_valor_fisico_sem_fonte_falha():
    s = _load_common()
    row = base_matriz(feature_source_mode="fila_extracao_necessaria")
    assert any("valor_fisico_sem_fonte" in e for e in s.validate_matriz_rows([row]))


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
    cal = base_calibracao(justificativa_tecnica="")
    assert any("calibracao_sem_justificativa" in e for e in s.validate_calibracao_rows([cal]))
    reav = {"canary_patch_id": "X", "justificativa_tecnica": ""}
    assert any("reavaliacao_sem_justificativa" in e for e in s.validate_reavaliacao_rows([reav]))
