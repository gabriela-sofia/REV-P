from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_protocolo_ground_truth_fail_closed_mv1 import (
    CAMINHO_DASHBOARD,
    CAMINHO_DOC_ANTI_LEAKAGE,
    CAMINHO_DOC_NEGATIVOS,
    CAMINHO_DOC_ONTOLOGIA,
    CAMINHO_ESTADOS,
    CAMINHO_GATES,
    CAMINHO_METRICAS,
    CAMINHO_RELATORIO,
    COLUNAS_DASHBOARD,
    COLUNAS_ESTADOS,
    COLUNAS_GATES,
    LINGUAGEM_PROIBIDA,
    executar,
)


def ler_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_protocolo_gera_todos_os_artefatos(tmp_path: Path) -> None:
    executar(tmp_path)

    for caminho in [
        CAMINHO_DOC_ONTOLOGIA,
        CAMINHO_DOC_NEGATIVOS,
        CAMINHO_DOC_ANTI_LEAKAGE,
        CAMINHO_ESTADOS,
        CAMINHO_GATES,
        CAMINHO_DASHBOARD,
        CAMINHO_RELATORIO,
        CAMINHO_METRICAS,
    ]:
        assert (tmp_path / caminho).exists(), caminho


def test_csvs_tem_colunas_obrigatorias(tmp_path: Path) -> None:
    executar(tmp_path)

    estados = ler_csv(tmp_path / CAMINHO_ESTADOS)
    gates = ler_csv(tmp_path / CAMINHO_GATES)
    dashboard = ler_csv(tmp_path / CAMINHO_DASHBOARD)

    assert list(estados[0].keys()) == COLUNAS_ESTADOS
    assert list(gates[0].keys()) == COLUNAS_GATES
    assert list(dashboard[0].keys()) == COLUNAS_DASHBOARD
    assert len(estados) == 7
    assert len(gates) == 9
    assert len(dashboard) == 11


def test_estados_fail_closed_nao_treinam(tmp_path: Path) -> None:
    executar(tmp_path)
    estados = {row["estado_label"]: row for row in ler_csv(tmp_path / CAMINHO_ESTADOS)}

    for nome in ["desconhecido", "bloqueado", "excluido"]:
        assert estados[nome]["pode_treinar"] == "false"

    assert estados["positivo_prata"]["pode_treinar"] == "false"
    assert estados["positivo_ouro"]["exige_evento_observado"] == "true"
    assert estados["positivo_ouro"]["exige_geometria"] == "true"
    assert estados["positivo_ouro"]["exige_janela_temporal_fechada"] == "true"
    assert estados["positivo_ouro"]["exige_revisao_humana"] == "true"


def test_negativos_exigem_evidencia_explicita(tmp_path: Path) -> None:
    executar(tmp_path)
    estados = {row["estado_label"]: row for row in ler_csv(tmp_path / CAMINHO_ESTADOS)}

    for nome in ["negativo_pareado", "negativo_dificil"]:
        texto = " ".join(estados[nome].values()).lower()
        assert "evidência explícita de não inundação" in texto
        assert "ausência" not in estados[nome]["evidencia_minima_exigida"].lower()

    dashboard = {row["componente"]: row for row in ler_csv(tmp_path / CAMINHO_DASHBOARD)}
    assert dashboard["negativos_formais"]["status_atual"] == "ausente"
    assert dashboard["negativos_formais"]["pode_liberar_treino"] == "false"


def test_json_statuses_permanecem_bloqueados_ou_ausentes(tmp_path: Path) -> None:
    metricas = executar(tmp_path)
    carregado = json.loads((tmp_path / CAMINHO_METRICAS).read_text(encoding="utf-8"))

    assert carregado == metricas
    assert carregado["treino_supervisionado_status"] == "bloqueado"
    assert carregado["ground_truth_operacional_status"] == "ausente"
    assert carregado["negativos_formais_status"] == "ausente"
    assert carregado["positivos_formais_status"] == "ausente"
    assert "G8_liberado_para_treino_true" in carregado["gates_bloqueados_atualmente"]


def test_relatorio_contem_guardrails(tmp_path: Path) -> None:
    executar(tmp_path)
    relatorio = (tmp_path / CAMINHO_RELATORIO).read_text(encoding="utf-8")

    for trecho in [
        "unknown nunca vira negativo",
        "Curitiba nunca vira negativo formal por default",
        "evidência contextual nunca vira label",
        "DINOv2 não prova inundação",
        "nenhum treino supervisionado é liberado",
    ]:
        assert trecho in relatorio


def test_sem_linguagem_proibida_em_campos_estruturados(tmp_path: Path) -> None:
    executar(tmp_path)
    textos = []
    for caminho in [CAMINHO_ESTADOS, CAMINHO_GATES, CAMINHO_DASHBOARD]:
        for row in ler_csv(tmp_path / caminho):
            textos.extend(str(valor).lower() for valor in row.values())
    textos.append((tmp_path / CAMINHO_METRICAS).read_text(encoding="utf-8").lower())
    combinado = "\n".join(textos)

    for termo in LINGUAGEM_PROIBIDA:
        assert termo.lower() not in combinado
