from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_integracao_final_marco_mv1_maturidade_revisao_humana import (
    CAMINHO_BLOQUEADORES,
    CAMINHO_EVIDENCIAS_GATES,
    CAMINHO_FILA,
    CAMINHO_MATURIDADE,
    CAMINHO_METRICAS,
    CAMINHO_RELATORIO,
    COLUNAS_BLOQUEADORES,
    COLUNAS_EVIDENCIAS_GATES,
    COLUNAS_FILA,
    COLUNAS_MATURIDADE,
    LINGUAGEM_PROIBIDA,
    executar,
)


def escrever_csv(path: Path, colunas: list[str], linhas: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=colunas, lineterminator="\n")
        writer.writeheader()
        writer.writerows(linhas)


def ler_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def montar_insumos_minimos(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("ref: refs/heads/marco/validacao-label-free-evidencia-estrutural-mv1\n", encoding="utf-8")

    metricas = tmp_path / "outputs_public/metrics"
    metricas.mkdir(parents=True)
    (metricas / "revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json").write_text(
        json.dumps(
            {
                "restauracao_v2dz_v2ef_status": "RESTAURACAO_FORENSE_FECHADA_SEM_GROUND_TRUTH_OPERACIONAL",
                "auditoria_temporal_status": "METADADOS_TEMPORAIS_INSUFICIENTES_TRILHA_B_RECOMENDADA",
                "validacao_label_free_status": "PILOTO_LABEL_FREE_EXPLORATORIO_EXECUTADO",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metricas / "revp_navegacao_downloads_evidencias_externas_mv1.json").write_text(
        json.dumps({"total_arquivos_baixados": 1, "total_geometrias_candidatas": 1, "total_eventos_candidatos": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (metricas / "revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.json").write_text(
        json.dumps({"itens_potenciais_para_revisao_humana": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (metricas / "revp_protocolo_ground_truth_fail_closed_mv1.json").write_text(
        json.dumps(
            {
                "status_protocolo": "PROTOCOLO_FAIL_CLOSED_FORMALIZADO_SEM_LIBERAR_TREINO",
                "treino_supervisionado_status": "bloqueado",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    escrever_csv(
        tmp_path / "outputs_public/tables/revp_indice_eventos_externos_candidatos_navegacao_mv1.csv",
        [
            "id_evento_externo",
            "id_fonte",
            "cidade",
            "regiao",
            "categoria_evento",
            "janela_temporal_status",
            "fonte_observacional_independente",
            "geometria_disponivel",
            "crs",
            "observacao",
        ],
        [
            {
                "id_evento_externo": "EV_TESTE_001",
                "id_fonte": "FONTE_TESTE",
                "cidade": "Recife",
                "regiao": "Nordeste",
                "categoria_evento": "desastre_hidrometeorologico_composto",
                "janela_temporal_status": "fechada",
                "fonte_observacional_independente": "sim",
                "geometria_disponivel": "sim",
                "crs": "EPSG:4326",
                "observacao": "Evento candidato para revisão futura.",
            }
        ],
    )
    escrever_csv(
        tmp_path / "outputs_public/tables/revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv",
        [
            "id_geometria",
            "id_fonte",
            "cidade",
            "regiao",
            "tipo_geometria",
            "crs",
            "representa_contexto",
            "representa_suscetibilidade",
            "representa_vulnerabilidade",
            "observacao",
        ],
        [
            {
                "id_geometria": "GEO_TESTE_001",
                "id_fonte": "FONTE_GEO",
                "cidade": "Curitiba",
                "regiao": "Sul",
                "tipo_geometria": "Polygon",
                "crs": "EPSG:4326",
                "representa_contexto": "true",
                "representa_suscetibilidade": "false",
                "representa_vulnerabilidade": "true",
                "observacao": "Geometria contextual para revisão.",
            }
        ],
    )
    escrever_csv(
        tmp_path / "outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv",
        [
            "id_arquivo",
            "id_fonte",
            "cidade",
            "regiao",
            "tipo_arquivo",
            "categoria_evento",
            "sha256_arquivo",
            "tem_geometria",
            "crs",
            "periodo_referencia_inicio",
            "periodo_referencia_fim",
            "observacao",
        ],
        [
            {
                "id_arquivo": "ARQ_TESTE_001",
                "id_fonte": "FONTE_ARQ",
                "cidade": "Petropolis",
                "regiao": "Sudeste",
                "tipo_arquivo": "relatorio_contextual",
                "categoria_evento": "contexto",
                "sha256_arquivo": "abc123",
                "tem_geometria": "false",
                "crs": "",
                "periodo_referencia_inicio": "",
                "periodo_referencia_fim": "",
                "observacao": "Arquivo contextual.",
            }
        ],
    )


def test_integracao_final_gera_todos_os_artefatos(tmp_path: Path) -> None:
    montar_insumos_minimos(tmp_path)
    executar(tmp_path)

    for caminho in [
        CAMINHO_RELATORIO,
        CAMINHO_MATURIDADE,
        CAMINHO_FILA,
        CAMINHO_EVIDENCIAS_GATES,
        CAMINHO_BLOQUEADORES,
        CAMINHO_METRICAS,
    ]:
        assert (tmp_path / caminho).exists(), caminho


def test_csvs_tem_colunas_obrigatorias(tmp_path: Path) -> None:
    montar_insumos_minimos(tmp_path)
    executar(tmp_path)

    assert list(ler_csv(tmp_path / CAMINHO_MATURIDADE)[0].keys()) == COLUNAS_MATURIDADE
    assert list(ler_csv(tmp_path / CAMINHO_FILA)[0].keys()) == COLUNAS_FILA
    assert list(ler_csv(tmp_path / CAMINHO_EVIDENCIAS_GATES)[0].keys()) == COLUNAS_EVIDENCIAS_GATES
    assert list(ler_csv(tmp_path / CAMINHO_BLOQUEADORES)[0].keys()) == COLUNAS_BLOQUEADORES


def test_json_statuses_obrigatorios(tmp_path: Path) -> None:
    montar_insumos_minimos(tmp_path)
    metricas = executar(tmp_path)
    carregado = json.loads((tmp_path / CAMINHO_METRICAS).read_text(encoding="utf-8"))

    assert carregado == metricas
    assert carregado["itens_podem_virar_label_agora"] == 0
    assert carregado["ground_truth_operacional_status"] == "ausente"
    assert carregado["treino_supervisionado_status"] == "bloqueado"
    assert carregado["decisao_final_mv1"] == "MARCO_MV1_CONSOLIDADO_REVIEW_ONLY_COM_FILA_DE_REVISAO_HUMANA_SEM_LIBERAR_TREINO"


def test_fila_nao_contem_label_liberado(tmp_path: Path) -> None:
    montar_insumos_minimos(tmp_path)
    executar(tmp_path)
    fila = ler_csv(tmp_path / CAMINHO_FILA)

    assert fila
    assert {row["pode_virar_label_agora"] for row in fila} == {"false"}
    assert all(row["pode_apoiar_ground_truth_futuro"] in {"true", "false"} for row in fila)


def test_matriz_maturidade_nao_libera_treino(tmp_path: Path) -> None:
    montar_insumos_minimos(tmp_path)
    executar(tmp_path)
    maturidade = ler_csv(tmp_path / CAMINHO_MATURIDADE)

    assert maturidade
    assert {row["pode_sustentar_treino_agora"] for row in maturidade} == {"false"}
    assert any(row["nivel_maturidade"] == "bloqueado_para_treino" for row in maturidade)


def test_relatorio_contem_guardrails(tmp_path: Path) -> None:
    montar_insumos_minimos(tmp_path)
    executar(tmp_path)
    relatorio = (tmp_path / CAMINHO_RELATORIO).read_text(encoding="utf-8")

    for trecho in [
        "nenhum modelo treinado",
        "ground truth operacional ausente",
        "DINOv2 usado apenas em leitura label-free",
        "landslide e flood permanecem separados",
        "fila de revisão humana não libera treino",
    ]:
        assert trecho in relatorio


def test_campos_estruturados_sem_linguagem_proibida(tmp_path: Path) -> None:
    montar_insumos_minimos(tmp_path)
    executar(tmp_path)
    textos = []
    for caminho in [CAMINHO_MATURIDADE, CAMINHO_FILA, CAMINHO_EVIDENCIAS_GATES, CAMINHO_BLOQUEADORES]:
        for row in ler_csv(tmp_path / caminho):
            textos.extend(str(valor).lower() for valor in row.values())
    textos.append((tmp_path / CAMINHO_METRICAS).read_text(encoding="utf-8").lower())
    combinado = "\n".join(textos)

    for termo in LINGUAGEM_PROIBIDA:
        assert termo.lower() not in combinado
