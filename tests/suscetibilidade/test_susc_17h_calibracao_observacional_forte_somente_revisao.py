"""Tests for SUSC-17H calibracao observacional forte somente revisao."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_17h_calibracao_observacional_forte_somente_revisao"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "preflight.json",
    OUT / "matriz_calibracao_observacional_forte.csv",
    OUT / "politica_componentes_calibracao_observacional.csv",
    OUT / "simulacao_pesos_calibracao_observacional.csv",
    OUT / "diagnostico_aderencia_observacional.csv",
    OUT / "gate_calibracao_observacional_forte.csv",
    OUT / "gate_prontidao_17b_pos_17h.csv",
    OUT / "resumo_por_canario.csv",
    OUT / "resumo_por_cenario.csv",
    OUT / "resumo_por_aderencia.csv",
    OUT / "summary.json",
    REPORTS / "SUSC_17H_CALIBRACAO_OBSERVACIONAL_FORTE_SOMENTE_REVISAO.md",
    SCHEMAS / "susc_17h_calibracao_observacional_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17h_calibracao_observacional_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17h_calibracao_common", path)
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
        "componente_fisico_topografico": "0.27", "componente_urbano_espectral": "0.44",
        "componente_umidade_espectral": "0.58", "componente_chuva_gatilho": "0.45",
        "componente_qualidade_evidencia": "0.74",
        "aderencia_observacional": "divergencia_fisica",
        "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false",
        "score_oficial": "false", "substituir_score_v6": "false", "usar_em_treino": "false",
        "not_ground_truth_reason": "indice observacional somente revisao",
    }
    row.update(overrides)
    return row


def base_simulacao(**overrides) -> dict:
    row = {
        "canary_patch_id": "S17C6_CANARY_REC_00001", "cenario": "cenario_base_v6_compativel",
        "peso_fisico": "0.40", "peso_urbano": "0.20", "peso_umidade": "0.10",
        "peso_chuva": "0.25", "peso_evidencia": "0.05",
        "score_oficial": "false", "substituir_score_v6": "false", "usar_em_treino": "false", "ground_truth": "false",
    }
    row.update(overrides)
    return row


# --- Build + validator + determinismo --------------------------------------
def test_build_validator_e_determinismo():
    result = run_script("build_susc_17h_calibracao_observacional_forte_somente_revisao.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    cards_first = {path: path.read_bytes() for path in sorted((OUT / "cartoes_calibracao").glob("*.md"))}
    assert cards_first
    result = run_script("build_susc_17h_calibracao_observacional_forte_somente_revisao.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    assert {path: path.read_bytes() for path in sorted((OUT / "cartoes_calibracao").glob("*.md"))} == cards_first
    result = run_script("validate_susc_17h_calibracao_observacional_forte_somente_revisao.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_matriz_integrada_com_5_canarios():
    matriz = read_csv(OUT / "matriz_calibracao_observacional_forte.csv")
    assert len(matriz) == 5
    for r in matriz:
        assert r["aderencia_observacional"]
        assert r["indice_observacional_somente_revisao"] not in {"", "not_available"}


def test_componentes_normalizados_0_1():
    matriz = read_csv(OUT / "matriz_calibracao_observacional_forte.csv")
    cols = [
        "componente_fisico_topografico", "componente_urbano_espectral",
        "componente_umidade_espectral", "componente_chuva_gatilho", "componente_qualidade_evidencia",
    ]
    for r in matriz:
        for c in cols:
            v = float(r[c])
            assert 0.0 <= v <= 1.0, f"{c}={v}"


def test_cenario_nao_oficial():
    sim = read_csv(OUT / "simulacao_pesos_calibracao_observacional.csv")
    assert sim
    assert all(r["score_oficial"] == "false" and r["substituir_score_v6"] == "false" for r in sim)
    # pesos somam positivo
    for r in sim:
        soma = sum(float(r[f]) for f in ["peso_fisico", "peso_urbano", "peso_umidade", "peso_chuva", "peso_evidencia"])
        assert soma > 0


def test_score_exploratorio_nao_substitui_score_v6():
    s = _load_common()
    row = base_matriz(substituir_score_v6="true")
    assert any("substituir_score_v6_true_proibido" in e for e in s.validate_matriz_rows([row]))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["score_v6_changed"] is False


def test_divergencia_fisica_preservada():
    s = _load_common()
    # fisico baixo, umidade/espectral alta -> divergencia fisica, nao forcar aderencia
    comp = {"fisico": 0.27, "urbano": 0.44, "umidade": 0.62, "chuva": 0.30, "evidencia": 0.74}
    aderencia, divergencias = s._adherence(comp, "low")
    assert aderencia == "divergencia_fisica"
    assert any("fisico" in d for d in divergencias)
    # e na matriz real ha divergencia preservada
    matriz = read_csv(OUT / "matriz_calibracao_observacional_forte.csv")
    assert any(r["aderencia_observacional"].startswith("divergencia") for r in matriz)


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
    assert summary["status_17b"] == "17B_AINDA_SEM_PRONTIDAO_BENCHMARK_AMOSTRA_LOCAL"


def test_componente_fora_de_0_1_falha():
    s = _load_common()
    row = base_matriz(componente_fisico_topografico="1.5")
    assert any("componente_fisico_topografico_fora_0_1" in e for e in s.validate_matriz_rows([row]))


def test_cenario_sem_pesos_positivos_falha():
    s = _load_common()
    row = base_simulacao(peso_fisico="0", peso_urbano="0", peso_umidade="0", peso_chuva="0", peso_evidencia="0")
    assert any("pesos_nao_somam_positivo" in e for e in s.validate_simulacao_rows([row]))


def test_diagnostico_sem_conclusao_falha():
    s = _load_common()
    row = {"canary_patch_id": "X", "status_aderencia": "divergencia_fisica", "conclusao_tecnica": ""}
    assert any("diagnostico_sem_conclusao" in e for e in s.validate_diagnostico_rows([row]))


def test_vocabulario_publico_proibido_falha():
    s = _load_common()
    hits = s.public_text_violations_text("texto publico com agente, LLM e IA")
    assert {"agente", "LLM", "IA"}.issubset(set(hits))


def test_status_invalido_falha():
    s = _load_common()
    row = base_matriz(aderencia_observacional="status_inexistente")
    assert any("aderencia_fora_enum" in e for e in s.validate_matriz_rows([row]))
