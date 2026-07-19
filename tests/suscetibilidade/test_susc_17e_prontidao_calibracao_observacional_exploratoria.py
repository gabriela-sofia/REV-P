"""Tests for SUSC-17E prontidao e calibracao observacional exploratoria."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_17e_prontidao_calibracao_observacional_exploratoria"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "preflight_prontidao_calibracao.csv",
    OUT / "preflight_prontidao_calibracao.json",
    OUT / "matriz_prontidao_calibracao.csv",
    OUT / "matriz_features_minimas.csv",
    OUT / "matriz_calibracao_exploratoria.csv",
    OUT / "hipoteses_ajuste_pesos_review_only.csv",
    OUT / "simulacao_sensibilidade_review_only.csv",
    OUT / "diagnostico_bloqueios_calibracao.csv",
    OUT / "recomendacoes_calibracao_futura.csv",
    OUT / "acoes_minimas_desbloqueio.csv",
    OUT / "gate_prontidao_calibracao_observacional.csv",
    OUT / "resumo_prontidao_por_status.csv",
    OUT / "resumo_prontidao_por_regiao.csv",
    OUT / "summary.json",
    REPORTS / "SUSC_17E_PRONTIDAO_CALIBRACAO_OBSERVACIONAL_EXPLORATORIA.md",
    SCHEMAS / "susc_17e_prontidao_calibracao_observacional_exploratoria_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17e_prontidao_calibracao_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17e_prontidao_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def base_prontidao(**overrides) -> dict:
    row = {
        "item_validacao_id": "S17E_TEST_0001",
        "candidate_event_id": "S17C_TEST",
        "source_id": "SRC_TEST",
        "geometry_id": "GEOM_TEST",
        "patch_id": "PATCH_TEST",
        "cidade": "Recife",
        "regiao": "REC",
        "tipo_fenomeno": "flood_inundation_alagamento",
        "data_evento": "2022-05-24..2022-05-30",
        "classe_vinculo": "exact_polygon_overlap",
        "qualidade_fonte": "85",
        "qualidade_temporal": "80",
        "qualidade_geometrica": "85",
        "qualidade_vinculo_patch": "65",
        "qualidade_fenomeno": "85",
        "qualidade_incerteza": "55",
        "disponibilidade_features": "50",
        "consistencia_features": "60",
        "estabilidade_regional": "35",
        "risco_vazamento_temporal": "80",
        "severidade_bloqueios": "55",
        "prontidao_calibracao": "pronto_para_calibracao_exploratoria_review_only",
        "grupos_features_disponiveis": "espectral;chuva_gatilho",
        "grupos_features_ausentes": "fisico;urbano_territorial",
        "accepted_for_evaluation": "true",
        "ground_truth": "false",
        "eligible_for_training": "false",
        "score_v7_allowed": "false",
        "review_only": "true",
        "justificativa_tecnica": "features espectrais e de chuva reais sustentam exploratoria review-only",
    }
    row.update(overrides)
    return row


def base_calibracao(**overrides) -> dict:
    row = {
        "item_validacao_id": "S17E_TEST_0001",
        "patch_id": "PATCH_TEST",
        "justificativa_tecnica": "justificativa preenchida",
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
    result = run_script("build_susc_17e_prontidao_calibracao_observacional_exploratoria.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    cards_first = {path: path.read_bytes() for path in sorted((OUT / "cartoes_prontidao").glob("*.md"))}
    assert cards_first
    result = run_script("build_susc_17e_prontidao_calibracao_observacional_exploratoria.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    assert {path: path.read_bytes() for path in sorted((OUT / "cartoes_prontidao").glob("*.md"))} == cards_first
    result = run_script("validate_susc_17e_prontidao_calibracao_observacional_exploratoria.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_caminho_funcional_exploratorio_entregue():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["items_count"] == 5
    assert summary["pronto_para_calibracao_exploratoria_review_only"] == 5
    assert summary["status_final_17e"] == "17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY"
    assert summary["caminho_funcional"] == "calibracao_observacional_exploratoria_review_only"
    assert summary["ground_truth"] is False
    assert summary["trainable"] is False
    assert summary["score_oficial"] is False
    assert summary["substituir_score_v6"] is False
    assert summary["score_v6_changed"] is False
    assert summary["score_v7_created"] is False


# --- Secao 12: casos exigidos ----------------------------------------------
def test_item_com_features_minimas_pode_virar_calibracao_forte():
    s = _load_common()
    quality = {
        "qualidade_incerteza": 85, "qualidade_temporal": 90, "qualidade_fenomeno": 85,
        "consistencia_features": 60, "risco_vazamento_temporal": 80,
    }
    decision = {"accepted_for_evaluation": "true", "confianca_tecnica_final": "alta"}
    geom = {"has_geometry_object": "true"}
    link = {"link_class": "exact_polygon_overlap"}
    groups = ["fisico", "espectral", "chuva_gatilho"]
    assert s._strong_ready(decision, geom, link, groups, quality) is True
    assert s._prontidao_status(True, True, groups) == "pronto_para_calibracao_forte"


def test_item_com_features_parciais_vira_exploratoria():
    s = _load_common()
    groups = ["espectral", "chuva_gatilho"]
    decision = {
        "accepted_for_evaluation": "true", "ground_truth": "false",
        "eligible_for_training": "false", "score_v7_allowed": "false",
        "patch_id": "PATCH", "geometry_id": "GEOM",
        "tipo_fenomeno": "flood_inundation_alagamento", "data_evento": "2022-05-24..2022-05-30",
    }
    geom = {"has_geometry_object": "true"}
    link = {"link_class": "exact_polygon_overlap"}
    assert s._exploratory_ready(decision, geom, link, groups) is True
    assert s._prontidao_status(False, True, groups) == "pronto_para_calibracao_exploratoria_review_only"


def test_item_sem_accepted_nao_pode_virar_calibracao():
    s = _load_common()
    groups = ["espectral", "chuva_gatilho"]
    decision = {
        "accepted_for_evaluation": "false", "ground_truth": "false",
        "eligible_for_training": "false", "score_v7_allowed": "false",
        "patch_id": "PATCH", "geometry_id": "GEOM",
        "tipo_fenomeno": "flood_inundation_alagamento", "data_evento": "2022-05-24..2022-05-30",
    }
    geom = {"has_geometry_object": "true"}
    link = {"link_class": "exact_polygon_overlap"}
    assert s._exploratory_ready(decision, geom, link, groups) is False
    quality = {
        "qualidade_incerteza": 85, "qualidade_temporal": 90, "qualidade_fenomeno": 85,
        "consistencia_features": 60, "risco_vazamento_temporal": 80,
    }
    assert s._strong_ready({**decision, "confianca_tecnica_final": "alta"}, geom, link,
                           ["fisico", "espectral"], quality) is False


def test_vazamento_temporal_critico_falha():
    s = _load_common()
    row = base_prontidao(prontidao_calibracao="pronto_para_calibracao_forte",
                         grupos_features_disponiveis="fisico;espectral;chuva_gatilho",
                         risco_vazamento_temporal="10")
    assert any("vazamento_temporal_critico" in e for e in s.validate_prontidao_rows([row]))


def test_score_exploratorio_nao_pode_ser_oficial():
    s = _load_common()
    row = base_calibracao(score_oficial="true")
    assert any("score_exploratorio_marcado_oficial" in e for e in s.validate_calibracao_rows([row]))


def test_score_exploratorio_nao_pode_substituir_score_v6():
    s = _load_common()
    row = base_calibracao(substituir_score_v6="true")
    assert any("substitui_score_v6" in e for e in s.validate_calibracao_rows([row]))


def test_ground_truth_treino_score_v7_proibidos():
    s = _load_common()
    row = base_prontidao(ground_truth="true", eligible_for_training="true", score_v7_allowed="true")
    errors = s.validate_prontidao_rows([row])
    assert any("ground_truth_true_proibido" in e for e in errors)
    assert any("eligible_for_training_true_proibido" in e for e in errors)
    assert any("score_v7_allowed_true_proibido" in e for e in errors)


def test_vocabulario_publico_proibido_falha():
    s = _load_common()
    hits = s.public_text_violations_text("texto publico com agente, LLM e IA")
    assert {"agente", "LLM", "IA"}.issubset(set(hits))


def test_status_invalido_falha():
    s = _load_common()
    row = base_prontidao(prontidao_calibracao="status_inexistente")
    assert any("status_prontidao_fora_enum" in e for e in s.validate_prontidao_rows([row]))


def test_justificativa_vazia_falha():
    s = _load_common()
    row = base_prontidao(justificativa_tecnica="")
    assert any("justificativa" in e for e in s.validate_prontidao_rows([row]))
    cal = base_calibracao(justificativa_tecnica="")
    assert any("calibracao_sem_justificativa" in e for e in s.validate_calibracao_rows([cal]))


def test_forte_sem_feature_fisica_falha():
    s = _load_common()
    row = base_prontidao(prontidao_calibracao="pronto_para_calibracao_forte",
                         grupos_features_disponiveis="espectral;chuva_gatilho")
    assert any("forte_sem_feature_fisica" in e for e in s.validate_prontidao_rows([row]))
