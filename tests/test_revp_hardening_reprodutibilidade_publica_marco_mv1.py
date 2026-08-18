from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ground_truth"))

from revp_hardening_reprodutibilidade_publica_marco_mv1 import (
    CAMINHO_CHECKLIST,
    CAMINHO_INDICE,
    CAMINHO_LOCAL_ONLY,
    CAMINHO_MANIFESTO,
    CAMINHO_METRICAS,
    CAMINHO_PLANO_PACOTE,
    CAMINHO_RELATORIO,
    COLUNAS_CHECKLIST,
    COLUNAS_INDICE,
    COLUNAS_LOCAL_ONLY,
    COLUNAS_MANIFESTO,
    COLUNAS_PLANO_PACOTE,
    FASES_PRINCIPAIS,
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


def montar_repo_minimo(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("ref: refs/heads/marco/validacao-label-free-evidencia-estrutural-mv1\n", encoding="utf-8")
    (tmp_path / "scripts/ground_truth").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    for caminho in [
        "scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py",
        "scripts/ground_truth/revp_validacao_label_free_evidencia_estrutural_mv1.py",
        "tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py",
        "tests/test_revp_validacao_label_free_evidencia_estrutural_mv1.py",
    ]:
        (tmp_path / caminho).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / caminho).write_text("# teste\n", encoding="utf-8")

    escrever_csv(
        tmp_path / "outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv",
        [
            "id_arquivo",
            "id_fonte",
            "cidade",
            "regiao",
            "instituicao",
            "titulo_arquivo",
            "tipo_arquivo",
            "formato",
            "url_pagina_origem",
            "url_final_download",
            "caminho_local_quarentena",
            "sha256_arquivo",
            "tamanho_bytes",
            "content_type",
            "data_acesso",
            "licenca_ou_uso",
            "periodo_referencia_inicio",
            "periodo_referencia_fim",
            "categoria_evento",
            "tem_geometria",
            "crs",
            "status_auditoria",
            "observacao",
        ],
        [
            {
                "id_arquivo": "ARQ_TESTE_001",
                "id_fonte": "FONTE_TESTE",
                "cidade": "Recife",
                "regiao": "Nordeste",
                "instituicao": "IBGE",
                "titulo_arquivo": "Arquivo contextual",
                "tipo_arquivo": "limite_municipal",
                "formato": "GeoJSON",
                "url_pagina_origem": "https://example.org/origem",
                "url_final_download": "https://example.org/arquivo.geojson",
                "caminho_local_quarentena": "local_only/evidencias_externas_quarentena/teste/arquivo.geojson",
                "sha256_arquivo": "a" * 64,
                "tamanho_bytes": "1234",
                "content_type": "application/geo+json",
                "data_acesso": "2026-06-19",
                "licenca_ou_uso": "dados_publicos_ibge",
                "periodo_referencia_inicio": "",
                "periodo_referencia_fim": "",
                "categoria_evento": "contexto",
                "tem_geometria": "true",
                "crs": "EPSG:4326",
                "status_auditoria": "aprovada_para_contexto",
                "observacao": "Contexto para revisao futura.",
            }
        ],
    )
    escrever_csv(
        tmp_path / "local_runs/dino_embeddings/v1ge/dino_expanded_embedding_manifest_v1ge.csv",
        ["patch_id", "dino_input_id", "region", "source_path", "embedding_path", "embedding_dim", "model_backbone", "device", "success", "failure_reason", "hash", "timestamp", "label_status", "target_status", "claim_scope"],
        [
            {
                "patch_id": "CUR_00038",
                "dino_input_id": "DINO_V1FU_SENTINEL_00001",
                "region": "Curitiba",
                "source_path": "C:/privado/nao_expor.tif",
                "embedding_path": "embeddings/DINO_V1FU_SENTINEL_00001.npz",
                "embedding_dim": "768",
                "model_backbone": "facebook/dinov2",
                "device": "cpu",
                "success": "SUCCESS",
                "failure_reason": "",
                "hash": "b" * 64,
                "timestamp": "2026-06-19T00:00:00+00:00",
                "label_status": "NO_LABEL",
                "target_status": "NO_TARGET",
                "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
            }
        ],
    )
    (tmp_path / "local_runs/dino_embeddings/v1ge/embeddings").mkdir(parents=True)
    (tmp_path / "local_runs/dino_embeddings/v1ge/embeddings/DINO_V1FU_SENTINEL_00001.npz").write_bytes(b"npz")
    escrever_csv(
        tmp_path / "manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv",
        ["dino_input_id", "canonical_patch_id", "region", "source_asset_id", "source_asset_type", "source_manifest", "asset_path_reference", "modality", "dino_scope", "eligibility_status", "split_group", "split_policy", "transform_policy_ref", "encoder_mode", "label_status", "target_status", "pixel_read_status", "claim_scope", "blocker_status", "notes"],
        [
            {
                "dino_input_id": "DINO_V1FU_SENTINEL_00001",
                "canonical_patch_id": "CUR_00038",
                "region": "Curitiba",
                "source_asset_id": "ASSET_001",
                "source_asset_type": "SENTINEL_TIF_ASSET",
                "source_manifest": "manifest.csv",
                "asset_path_reference": "data/sentinel/patch.tif",
                "modality": "sentinel_raster_path_only",
                "dino_scope": "SENTINEL_FIRST_EMBEDDING_INPUT",
                "eligibility_status": "READY_SENTINEL_FIRST_REVIEW_ONLY",
                "split_group": "grupo",
                "split_policy": "sem_random_split",
                "transform_policy_ref": "policy.csv",
                "encoder_mode": "frozen_encoder",
                "label_status": "NO_LABEL",
                "target_status": "NO_TARGET",
                "pixel_read_status": "NOT_READ",
                "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
                "blocker_status": "NONE",
                "notes": "metadata only",
            }
        ],
    )


def test_gera_artefatos_publicos_obrigatorios(tmp_path: Path) -> None:
    montar_repo_minimo(tmp_path)
    executar(tmp_path)

    for caminho in [CAMINHO_RELATORIO, CAMINHO_MANIFESTO, CAMINHO_INDICE, CAMINHO_LOCAL_ONLY, CAMINHO_PLANO_PACOTE, CAMINHO_CHECKLIST, CAMINHO_METRICAS]:
        assert (tmp_path / caminho).exists(), caminho


def test_csvs_tem_colunas_obrigatorias(tmp_path: Path) -> None:
    montar_repo_minimo(tmp_path)
    executar(tmp_path)

    assert list(ler_csv(tmp_path / CAMINHO_MANIFESTO)[0].keys()) == COLUNAS_MANIFESTO
    assert list(ler_csv(tmp_path / CAMINHO_INDICE)[0].keys()) == COLUNAS_INDICE
    assert list(ler_csv(tmp_path / CAMINHO_LOCAL_ONLY)[0].keys()) == COLUNAS_LOCAL_ONLY
    assert list(ler_csv(tmp_path / CAMINHO_PLANO_PACOTE)[0].keys()) == COLUNAS_PLANO_PACOTE
    assert list(ler_csv(tmp_path / CAMINHO_CHECKLIST)[0].keys()) == COLUNAS_CHECKLIST


def test_json_statuses_fixos_conservadores(tmp_path: Path) -> None:
    montar_repo_minimo(tmp_path)
    metricas = executar(tmp_path)
    carregado = json.loads((tmp_path / CAMINHO_METRICAS).read_text(encoding="utf-8"))

    assert carregado == metricas
    assert carregado["brutos_pesados_em_outputs_public"] == 0
    assert carregado["pode_reproduzir_marco_integral_so_com_git"] is False
    assert carregado["pode_verificar_marco_por_hashes_e_indices"] is True
    assert carregado["pode_fazer_stage_seletivo"] == "true_com_revisao_manual"
    assert carregado["pode_usar_git_add_A"] is False
    assert carregado["ground_truth_operacional_status"] == "ausente"
    assert carregado["treino_supervisionado_status"] == "bloqueado"
    assert carregado["itens_podem_virar_label_agora"] == 0


def test_manifesto_contem_fases_principais(tmp_path: Path) -> None:
    montar_repo_minimo(tmp_path)
    executar(tmp_path)
    fases = {row["fase"] for row in ler_csv(tmp_path / CAMINHO_MANIFESTO)}

    for fase in FASES_PRINCIPAIS:
        assert fase in fases


def test_local_only_e_dependencia_nao_publica_git(tmp_path: Path) -> None:
    montar_repo_minimo(tmp_path)
    executar(tmp_path)
    deps = ler_csv(tmp_path / CAMINHO_LOCAL_ONLY)
    manifesto = ler_csv(tmp_path / CAMINHO_MANIFESTO)

    assert deps
    assert {row["pode_ser_publicado_no_git"] for row in deps} == {"false"}
    assert any(row["eh_local_only"] == "true" and row["eh_publico_git"] == "false" for row in manifesto)


def test_plano_externo_inclui_embeddings_indices_e_geotiff_bloqueado(tmp_path: Path) -> None:
    montar_repo_minimo(tmp_path)
    executar(tmp_path)
    plano = ler_csv(tmp_path / CAMINHO_PLANO_PACOTE)
    itens = {row["item_pacote"] for row in plano}

    assert "DINO_V1FU_SENTINEL_00001" in itens
    assert "MANIFESTO_DINO_LOCAL_V1GE" in itens
    assert "raw_checksums_e_indices_publicos" in itens
    assert "sentinel_scene_asset_ids_mv1" in itens
    assert "geotiffs_pesados_sentinel" in itens
    assert any(row["acao_antes_publicar"].startswith("Nao incluir") for row in plano if row["item_pacote"] == "geotiffs_pesados_sentinel")


def test_relatorio_contem_instrucoes_reproducao(tmp_path: Path) -> None:
    montar_repo_minimo(tmp_path)
    executar(tmp_path)
    texto = (tmp_path / CAMINHO_RELATORIO).read_text(encoding="utf-8")

    assert "## 10. Como reproduzir scripts principais" in texto
    assert "python scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py --repo-root ." in texto
    assert "git add <lista explicita>" in texto


def test_campos_estruturados_sem_linguagem_proibida(tmp_path: Path) -> None:
    montar_repo_minimo(tmp_path)
    executar(tmp_path)
    textos = []
    for caminho in [CAMINHO_MANIFESTO, CAMINHO_INDICE, CAMINHO_LOCAL_ONLY, CAMINHO_PLANO_PACOTE, CAMINHO_CHECKLIST]:
        for row in ler_csv(tmp_path / caminho):
            textos.extend(str(valor).lower() for valor in row.values())
    textos.append((tmp_path / CAMINHO_METRICAS).read_text(encoding="utf-8").lower())
    combinado = "\n".join(textos)

    for termo in LINGUAGEM_PROIBIDA:
        assert termo.lower() not in combinado
