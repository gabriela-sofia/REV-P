from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_validacao_label_free_evidencia_estrutural_mv1 import executar


def escrever_csv(path: Path, colunas: list[str], linhas: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=colunas, lineterminator="\n")
        writer.writeheader()
        writer.writerows(linhas)


def ler_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def montar_repo_com_12_embeddings(tmp_path: Path) -> None:
    patches = [f"CUR_{i:05d}" for i in range(4)] + [f"PET_{i:05d}" for i in range(4)] + [f"REC_{i:05d}" for i in range(4)]
    cidades = {patch: ("Curitiba" if patch.startswith("CUR") else "Petrópolis" if patch.startswith("PET") else "Recife") for patch in patches}
    escrever_csv(
        tmp_path / "outputs_public/tables/table_dino_embedding_inventory.csv",
        ["patch_id", "dino_input_id", "region", "embedding_dim", "model_backbone", "vector_sha256", "vector_norm_l2", "review_only", "label_status", "target_status", "raw_embedding_publication", "allowed_use", "methodological_note"],
        [
            {
                "patch_id": patch,
                "dino_input_id": f"DINO_{idx:04d}",
                "region": cidades[patch],
                "embedding_dim": "768",
                "model_backbone": "facebook/dinov2-with-registers-base",
                "vector_sha256": f"hash_{idx}",
                "vector_norm_l2": "1.0",
                "review_only": "True",
                "label_status": "NO_LABEL",
                "target_status": "SEM_ALVO_SUPERVISIONADO",
                "raw_embedding_publication": "NOT_PUBLISHED_LOCAL_ONLY",
                "allowed_use": "similarity|review_prioritization",
                "methodological_note": "review-only",
            }
            for idx, patch in enumerate(patches)
        ],
    )
    matriz_linhas = []
    for patch_a in patches:
        row = {"patch_id": patch_a}
        for patch_b in patches:
            if patch_a == patch_b:
                sim = 1.0
            elif cidades[patch_a] == cidades[patch_b]:
                sim = 0.9
            else:
                sim = 0.6
            row[patch_b] = str(sim)
        matriz_linhas.append(row)
    escrever_csv(tmp_path / "outputs_public/tables/table_dino_similarity_matrix.csv", ["patch_id", *patches], matriz_linhas)
    escrever_csv(
        tmp_path / "local_runs/dino_embeddings/v1gv/evidence_coverage_matrix_v1gv.csv",
        ["canonical_patch_id", "region", "terrain_geocuritiba", "land_use", "drainage", "population_density", "administrative_defesa_civil", "terrain_sgb_rigeo", "land_use_fbds", "geological_cprm", "terrain_pe3d", "drainage_esig", "coastal_context"],
        [{"canonical_patch_id": patch, "region": cidades[patch], "terrain_geocuritiba": "PARTIAL", "land_use": "MISSING", "drainage": "MISSING", "population_density": "MISSING", "administrative_defesa_civil": "MISSING", "terrain_sgb_rigeo": "", "land_use_fbds": "", "geological_cprm": "", "terrain_pe3d": "", "drainage_esig": "", "coastal_context": ""} for patch in patches],
    )


def test_mv1_gera_piloto_label_free_com_12_embeddings(tmp_path: Path) -> None:
    montar_repo_com_12_embeddings(tmp_path)

    metricas = executar(tmp_path)

    assert metricas["total_embeddings_encontrados"] == 12
    assert metricas["total_embeddings_validos"] == 12
    assert metricas["decisao_metodologica"] == "PILOTO_LABEL_FREE_COM_LIMITACAO_AMOSTRAL"
    assert metricas["trilha_b_status"] == "ASSUMIDA_COMO_EIXO_CIENTIFICO_IMEDIATO"

    principal = ler_csv(tmp_path / "outputs_public/tables/revp_validacao_label_free_evidencia_estrutural_mv1.csv")
    assert len(principal) == 12
    assert {row["score_evidencia_eh_label"] for row in principal} == {"false"}
    assert {row["status_amostra"] for row in principal} == {"PILOTO_EXPLORATORIO_LABEL_FREE"}


def test_mv1_preserva_guardrails_publicos(tmp_path: Path) -> None:
    montar_repo_com_12_embeddings(tmp_path)
    executar(tmp_path)

    guardrails = ler_csv(tmp_path / "outputs_public/tables/revp_guardrails_validacao_label_free_mv1.csv")
    nomes = {row["guardrail"] for row in guardrails}

    assert "sem treino supervisionado" in nomes
    assert "sem label binário" in nomes
    assert "Curitiba não vira negativo formal" in nomes
    assert "evidência administrativa não vira label" in nomes
    assert {row["status"] for row in guardrails} == {"PRESERVADO"}


def test_mv1_bloqueia_sem_embeddings_rastreaveis(tmp_path: Path) -> None:
    metricas = executar(tmp_path)

    assert metricas["decisao_metodologica"] == "BLOQUEADO_SEM_EMBEDDINGS_RASTREAVEIS"
    assert metricas["total_embeddings_encontrados"] == 0
    assert json.loads((tmp_path / "outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json").read_text(encoding="utf-8"))["total_embeddings_validos"] == 0
