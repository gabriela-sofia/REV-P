from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ground_truth"))

from revp_auditoria_prontidao_temporal_assets_mv1 import CSV_COLUNAS, executar


def escrever_csv(path: Path, colunas: list[str], linhas: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=colunas, lineterminator="\n")
        writer.writeheader()
        writer.writerows(linhas)


def ler_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_auditoria_gera_saidas_publicas_em_portugues(tmp_path: Path) -> None:
    manifesto = tmp_path / "manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv"
    escrever_csv(
        manifesto,
        ["dino_input_id", "canonical_patch_id", "region", "source_asset_id"],
        [
            {"dino_input_id": "DINO_1", "canonical_patch_id": "CUR_0001", "region": "Curitiba", "source_asset_id": "ASSET_1"},
            {"dino_input_id": "DINO_2", "canonical_patch_id": "CUR_0002", "region": "Curitiba", "source_asset_id": "ASSET_2"},
        ],
    )
    qualidade = tmp_path / "datasets/official_anchor_sentinel_patch_quality_registry.csv"
    escrever_csv(
        qualidade,
        ["quality_id", "reference_patch_id", "scene_date", "local_cloud_fraction", "cloud_metadata_global"],
        [
            {"quality_id": "Q1", "reference_patch_id": "REF_1", "scene_date": "2022-01-01", "local_cloud_fraction": "0.1", "cloud_metadata_global": "5"},
        ],
    )

    metricas = executar(tmp_path)

    tabela = tmp_path / "outputs_public/tables/revp_auditoria_prontidao_temporal_assets_mv1.csv"
    relatorio = tmp_path / "outputs_public/execution_reports/revp_auditoria_prontidao_temporal_assets_mv1.md"
    json_path = tmp_path / "outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json"

    assert tabela.exists()
    assert relatorio.exists()
    assert json_path.exists()
    assert ler_csv(tabela)[0].keys() == set(CSV_COLUNAS)
    assert metricas["decisao_global"] == "METADADOS_TEMPORAIS_INSUFICIENTES_TRILHA_B_RECOMENDADA"

    conteudo = json.loads(json_path.read_text(encoding="utf-8"))
    for chave in [
        "total_patches",
        "total_assets",
        "contagem_por_regiao",
        "patches_com_1_data",
        "patches_com_2_datas",
        "patches_com_3_ou_mais_datas",
        "patches_elegiveis_deslocamento_temporal",
        "decisao_global",
        "bloqueios_principais",
        "guardrails_preservados",
    ]:
        assert chave in conteudo


def test_trilha_a_quando_ha_30_patches_com_3_datas_e_nuvem(tmp_path: Path) -> None:
    linhas = []
    for idx in range(30):
        for data_idx in range(3):
            linhas.append(
                {
                    "quality_id": f"Q{idx}_{data_idx}",
                    "reference_patch_id": f"PATCH_{idx:03d}",
                    "scene_date": f"2022-01-{data_idx + 1:02d}",
                    "local_cloud_fraction": "0.0",
                    "cloud_metadata_global": "5.0",
                }
            )
    escrever_csv(
        tmp_path / "datasets/official_anchor_sentinel_patch_quality_registry.csv",
        ["quality_id", "reference_patch_id", "scene_date", "local_cloud_fraction", "cloud_metadata_global"],
        linhas,
    )

    metricas = executar(tmp_path)

    assert metricas["patches_elegiveis_deslocamento_temporal"] == 30
    assert metricas["decisao_global"] == "TRILHA_A_DESLOCAMENTO_TEMPORAL_EMBEDDINGS"


def test_guardrails_nao_promovem_ground_truth_labels_ou_treino(tmp_path: Path) -> None:
    escrever_csv(
        tmp_path / "datasets/protocolo_c/v2ap_sentinel_asset_inventory.csv",
        ["asset_id", "path", "region_detected", "patch_id_detected", "date_detected", "safe_to_use_as_crosswalk_evidence"],
        [
            {
                "asset_id": "A1",
                "path": "asset.tif",
                "region_detected": "Recife",
                "patch_id_detected": "REC_001",
                "date_detected": "2022-05-24",
                "safe_to_use_as_crosswalk_evidence": "true",
            }
        ],
    )

    metricas = executar(tmp_path)

    assert "sem_treino_de_modelo" in metricas["guardrails_preservados"]
    assert "sem_criacao_de_labels" in metricas["guardrails_preservados"]
    assert "sem_negativos_formais" in metricas["guardrails_preservados"]
    assert "sem_promocao_de_ground_truth" in metricas["guardrails_preservados"]
