"""Tests for SUSC-17I ampliacao regional da amostra observacional."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_17i_ampliacao_regional_amostra_observacional"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "preflight.json",
    OUT / "inventario_regional_eventos_observacionais.csv",
    OUT / "matriz_amostra_observacional_expandida.csv",
    OUT / "matriz_disponibilidade_features_regionais.csv",
    OUT / "fila_regional_desbloqueio_observacional.csv",
    OUT / "gate_ampliacao_regional_observacional.csv",
    OUT / "gate_prontidao_17b_pos_17i.csv",
    OUT / "resumo_por_regiao.csv",
    OUT / "resumo_por_status.csv",
    OUT / "resumo_por_fenomeno.csv",
    OUT / "summary.json",
    REPORTS / "SUSC_17I_AMPLIACAO_REGIONAL_AMOSTRA_OBSERVACIONAL.md",
    SCHEMAS / "susc_17i_ampliacao_regional_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17i_ampliacao_regional_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17i_ampliacao_common", path)
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


def cand(**overrides) -> dict:
    c = {
        "candidate_event_id": "S17C_TEST", "region": "CUR", "city": "Curitiba",
        "event_date_candidate": "2022-01-15..2022-01-16", "event_date_precision": "range",
        "phenomenon_type": "flood_inundation_alagamento", "source_type": "official_observed_event_polygon",
        "authority_tier": "official", "geometry_status": "resolved_geometry_object",
        "has_address": "false", "has_point": "true",
    }
    c.update(overrides)
    return c


def base_matriz(**overrides) -> dict:
    row = {
        "item_amostra_id": "S17I_ITEM_0001", "candidate_event_id": "S17C_TEST",
        "regiao": "CUR", "cidade": "Curitiba", "data_evento": "2022-01-15..2022-01-16",
        "tipo_fenomeno": "flood_inundation_alagamento", "classe_fenomeno": "inundacao",
        "geometry_id": "GEOM_X", "patch_id": "PATCH_X", "classe_vinculo": "exact_polygon_overlap",
        "status_prontidao": "candidato_observacional_parcial", "uso_permitido": "avaliacao_somente_revisao",
        "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false",
        "not_ground_truth_reason": "regional review-only", "justificativa_tecnica": "candidato parcial",
    }
    row.update(overrides)
    return row


# --- Build + validator + determinismo --------------------------------------
def test_build_validator_e_determinismo():
    result = run_script("build_susc_17i_ampliacao_regional_amostra_observacional.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    cards_first = {path: path.read_bytes() for path in sorted((OUT / "cartoes_regionais").glob("*.md"))}
    assert cards_first
    result = run_script("build_susc_17i_ampliacao_regional_amostra_observacional.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    assert {path: path.read_bytes() for path in sorted((OUT / "cartoes_regionais").glob("*.md"))} == cards_first
    result = run_script("validate_susc_17i_ampliacao_regional_amostra_observacional.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_matriz_regional_criada_e_status():
    matriz = read_csv(OUT / "matriz_amostra_observacional_expandida.csv")
    assert len(matriz) == 41  # 11 Curitiba + 30 Petropolis
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["status_final_17i"] in {
        "17I_AMOSTRA_REGIONAL_EXPANDIDA_COM_CANDIDATOS_FORTES",
        "17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS",
        "17I_CURITIBA_PETROPOLIS_SEM_PRONTIDAO_MAS_COM_PLANO",
        "17I_APROXIMACAO_17B_SEM_BENCHMARK",
    }
    assert summary["score_v6_changed"] is False


# --- Secao 14: casos exigidos ----------------------------------------------
def test_curitiba_com_geometria_forte_vira_forte():
    s = _load_common()
    assert s._classify_status(cand(region="CUR", geometry_status="resolved_geometry_object")) == "candidato_observacional_forte"


def test_petropolis_com_fenomeno_misto_fica_bloqueado():
    s = _load_common()
    st = s._classify_status(cand(region="PET", phenomenon_type="hydrometeorological_or_unknown",
                                 source_type="official_observed_event_point", geometry_status="official_point_or_coordinate_unvalidated"))
    assert st == "bloqueado_por_fenomeno_misto"
    # e o validator barra misto marcado como inundacao forte/parcial
    row = base_matriz(classe_fenomeno="misto", status_prontidao="candidato_observacional_parcial")
    assert any("fenomeno_misto_como_inundacao" in e for e in s.validate_matriz_rows([row]))


def test_deslizamento_nao_vira_inundacao():
    s = _load_common()
    assert s._classify_status(cand(phenomenon_type="mass_movement")) == "bloqueado_por_fenomeno_misto"


def test_bairro_textual_nao_vira_geometria_forte():
    s = _load_common()
    c = cand(geometry_status="address_text_only", has_address="true", has_point="false", source_type="official_address_resolved")
    assert s._strong_geometry(c) is False
    assert s._classify_status(c) == "bloqueado_sem_geometria"


def test_candidato_sem_data_bloqueado():
    s = _load_common()
    assert s._classify_status(cand(event_date_candidate="not_available", event_date_precision="unknown",
                                   geometry_status="documentary_or_administrative_only")) == "bloqueado_sem_data"


def test_candidato_sem_geometria_bloqueado():
    s = _load_common()
    # inundacao datada, fonte fraca, sem geometria forte -> bloqueado_sem_geometria
    st = s._classify_status(cand(source_type="insufficient", authority_tier="internal_registry",
                                 geometry_status="no_strong_geometry"))
    assert st == "bloqueado_sem_geometria"


def test_ground_truth_treino_score_v7_proibidos():
    s = _load_common()
    row = base_matriz(ground_truth="true", eligible_for_training="true", score_v7_allowed="true")
    errors = s.validate_matriz_rows([row])
    assert any("ground_truth_true_proibido" in e for e in errors)
    assert any("eligible_for_training_true_proibido" in e for e in errors)
    assert any("score_v7_allowed_true_proibido" in e for e in errors)


def test_17b_nao_criado():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_17b_criado"] is False
    assert summary["status_17b"] in {
        "17B_AINDA_BLOQUEADO_AMOSTRA_INSUFICIENTE",
        "17B_APROXIMACAO_COM_AMOSTRA_REGIONAL_PARCIAL",
    }


def test_candidato_forte_sem_data_falha_no_validator():
    s = _load_common()
    row = base_matriz(status_prontidao="candidato_observacional_forte", data_evento="not_available")
    assert any("forte_sem_data" in e for e in s.validate_matriz_rows([row]))


def test_vocabulario_publico_proibido_falha():
    s = _load_common()
    hits = s.public_text_violations_text("texto publico com agente, LLM e IA")
    assert {"agente", "LLM", "IA"}.issubset(set(hits))


def test_status_invalido_falha():
    s = _load_common()
    row = base_matriz(status_prontidao="status_inexistente")
    assert any("status_fora_enum" in e for e in s.validate_matriz_rows([row]))


def test_justificativa_vazia_falha():
    s = _load_common()
    row = base_matriz(justificativa_tecnica="")
    assert any("justificativa_vazia" in e for e in s.validate_matriz_rows([row]))
