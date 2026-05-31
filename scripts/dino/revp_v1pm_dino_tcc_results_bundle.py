"""REV-P v1pm — DINO TCC results bundle.

Consolidates v1pg-v1pl into TCC-ready tables, a manifest and a scientific
summary. Does not recompute analyses — reads their outputs and re-frames them
for writing. Final DINO status is READY_REVIEW_ONLY only if valid embeddings
exist, otherwise NOT_FOUND_FAIL_CLOSED.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, read_csv, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

IN_DISCOVERY_S = _p("REVP_V1PM_IN_DISCOVERY_S", DATASETS / "dino_artifact_discovery_summary_v1pg.csv")
IN_REGISTRY = _p("REVP_V1PM_IN_REGISTRY", DATASETS / "dino_embedding_feature_store_registry_v1ph.csv")
IN_REGISTRY_S = _p("REVP_V1PM_IN_REGISTRY_S", DATASETS / "dino_embedding_feature_store_summary_v1ph.csv")
IN_NEIGHBORS = _p("REVP_V1PM_IN_NEIGHBORS", DATASETS / "dino_similarity_neighbors_v1pj.csv")
IN_SIM_S = _p("REVP_V1PM_IN_SIM_S", DATASETS / "dino_similarity_summary_v1pj.csv")
IN_PCA = _p("REVP_V1PM_IN_PCA", DATASETS / "dino_pca_projection_v1pk.csv")
IN_CLUSTER = _p("REVP_V1PM_IN_CLUSTER", DATASETS / "dino_cluster_exploratory_v1pk.csv")
IN_PCA_S = _p("REVP_V1PM_IN_PCA_S", DATASETS / "dino_pca_cluster_summary_v1pk.csv")
IN_CROSSWALK = _p("REVP_V1PM_IN_CROSSWALK", DATASETS / "dino_protocol_c_crosswalk_v1pl.csv")
IN_CROSSWALK_S = _p("REVP_V1PM_IN_CROSSWALK_S", DATASETS / "dino_protocol_c_crosswalk_summary_v1pl.csv")

OUT_T_EMB = _p("REVP_V1PM_OUT_T_EMB", DATASETS / "dino_tcc_table_embedding_inventory_v1pm.csv")
OUT_T_SIM = _p("REVP_V1PM_OUT_T_SIM", DATASETS / "dino_tcc_table_similarity_results_v1pm.csv")
OUT_T_PCA = _p("REVP_V1PM_OUT_T_PCA", DATASETS / "dino_tcc_table_pca_cluster_results_v1pm.csv")
OUT_T_XW = _p("REVP_V1PM_OUT_T_XW", DATASETS / "dino_tcc_table_protocol_c_crosswalk_v1pm.csv")
OUT_MANIFEST = _p("REVP_V1PM_OUT_MANIFEST", DATASETS / "dino_tcc_results_manifest_v1pm.csv")
OUT_SUMMARY = _p("REVP_V1PM_OUT_SUMMARY", DATASETS / "dino_tcc_results_scientific_summary_v1pm.csv")
SCHEMA_DIR = _p("REVP_V1PM_SCHEMA_DIR", SCHEMAS)
DOC = _p("REVP_V1PM_DOC", DOCS / "revp_v1pm_dino_tcc_results_bundle.md")

T_EMB_FIELDS = ["embedding_id", "patch_id", "region", "vector_dim", "embedding_status", "dino_allowed_use", "is_duplicate_vector"]
T_SIM_FIELDS = ["query_patch_id", "neighbor_rank", "neighbor_patch_id", "cosine_similarity", "same_region", "representation_use"]
T_PCA_FIELDS = ["patch_id", "region", "pca_x", "pca_y", "exploratory_cluster_id", "cluster_use", "can_be_used_as_class"]
T_XW_FIELDS = ["crosswalk_id", "patch_id", "embedding_status", "protocol_c_event_id", "protocol_c_candidate_level", "temporal_status", "dino_can_validate_event"]
MANIFEST_FIELDS = ["artifact_id", "stage", "relative_path", "rows", "header_present", "role"]
SUMMARY_FIELDS = ["summary_id", "metric", "value", "interpretation", "methodological_status", "writing_use"]

TCC_TEXT = (
    "Os embeddings DINOv2 foram tratados como representação vetorial auto-supervisionada "
    "dos patches Sentinel, não como rótulo supervisionado. As análises de similaridade, "
    "vizinhança, PCA e agrupamento exploratório foram usadas para avaliar coerência "
    "visual/semântica entre patches e regiões, sem validar evento observado, sem criar "
    "ground truth operacional e sem treinar classificador de inundação."
)


def _stat(path: Path, key: str) -> str:
    for r in read_csv(path):
        if r.get("stat_key") == key:
            return r.get("stat_value", "0")
    return "0"


def _project(rows: list[dict[str, str]], fields: list[str]) -> list[dict[str, Any]]:
    return [{f: r.get(f, "") for f in fields} for r in rows]


def build_summary() -> tuple[list[dict[str, Any]], str]:
    artifacts = _stat(IN_DISCOVERY_S, "artifacts_scanned")
    likely_emb = _stat(IN_DISCOVERY_S, "likely_embedding_artifacts")
    parsed = _stat(IN_REGISTRY_S, "embeddings_parsed")
    valid = _stat(IN_REGISTRY_S, "valid_768d_embeddings")
    invalid = _stat(IN_REGISTRY_S, "invalid_embeddings")
    dups = _stat(IN_REGISTRY_S, "duplicate_vectors")
    neighbors = _stat(IN_SIM_S, "neighbor_pairs_generated")
    pca_rows = _stat(IN_PCA_S, "pca_rows_generated")
    clusters = _stat(IN_PCA_S, "exploratory_clusters_generated")
    xw_rows = _stat(IN_CROSSWALK_S, "crosswalk_rows")

    final = "DINO_REPRESENTATION_LAYER_READY_REVIEW_ONLY" if valid not in ("0", "") else "DINO_EMBEDDINGS_NOT_FOUND_FAIL_CLOSED"

    def s(i: int, m: str, v: str, interp: str, status: str = "RESULTADO_FINAL", use: str = "resultado_negativo_auditavel") -> dict[str, Any]:
        return {"summary_id": f"V1PM_S{i:03d}", "metric": m, "value": v,
                "interpretation": interp, "methodological_status": status, "writing_use": use}

    rows = [
        s(1, "artifacts_scanned", artifacts, "Artefatos do repositório escaneados (metadata-only)", "AUDITAVEL", "metodologia_auditoria"),
        s(2, "likely_embedding_artifacts", likely_emb, "Artefatos prováveis de embedding por termos", "AUDITAVEL", "metodologia_auditoria"),
        s(3, "embeddings_parsed", parsed, "Vetores de embedding parseados de fontes reais", "AUDITAVEL", "metodologia_auditoria"),
        s(4, "valid_768d_embeddings", valid, "Embeddings válidos de 768D em regime review-only"),
        s(5, "invalid_embeddings", invalid, "Embeddings inválidos (dim/NaN/inf/zero) bloqueados"),
        s(6, "duplicate_vectors", dups, "Vetores duplicados (flag de revisão, nunca label)"),
        s(7, "neighbor_pairs_generated", neighbors, "Pares de vizinhança por similaridade exploratória", "AUDITAVEL", "metodologia_auditoria"),
        s(8, "pca_rows_generated", pca_rows, "Linhas de projeção PCA exploratória", "AUDITAVEL", "metodologia_auditoria"),
        s(9, "exploratory_clusters_generated", clusters, "Clusters exploratórios (não são classe)", "AUDITAVEL", "metodologia_auditoria"),
        s(10, "protocol_c_crosswalk_rows", xw_rows, "Linhas de crosswalk DINO ↔ Protocolo C", "AUDITAVEL", "metodologia_auditoria"),
        s(11, "labels_created", "0", "Labels operacionais criadas — 0 por design"),
        s(12, "training_targets_created", "0", "Targets de treinamento criados — 0 por design"),
        s(13, "final_dino_status", final, "Status final da camada DINO review-only", "RESULTADO_FINAL", "conclusao_auditavel"),
    ]
    return rows, final


def run() -> None:
    t_emb = _project(read_csv(IN_REGISTRY), T_EMB_FIELDS)
    t_sim = _project(read_csv(IN_NEIGHBORS), T_SIM_FIELDS)
    pca_rows = read_csv(IN_PCA)
    cluster_by_patch = {r.get("patch_id"): r for r in read_csv(IN_CLUSTER)}
    t_pca: list[dict[str, Any]] = []
    for r in pca_rows:
        c = cluster_by_patch.get(r.get("patch_id"), {})
        t_pca.append({
            "patch_id": r.get("patch_id", ""), "region": r.get("region", ""),
            "pca_x": r.get("pca_x", ""), "pca_y": r.get("pca_y", ""),
            "exploratory_cluster_id": c.get("exploratory_cluster_id", ""),
            "cluster_use": c.get("cluster_use", ""),
            "can_be_used_as_class": c.get("can_be_used_as_class", "false"),
        })
    t_xw = _project(read_csv(IN_CROSSWALK), T_XW_FIELDS)
    summary, final = build_summary()

    manifest = [
        {"artifact_id": "V1PM_T001", "stage": "v1pm", "relative_path": OUT_T_EMB.name, "rows": str(len(t_emb)), "header_present": "true", "role": "embedding_inventory"},
        {"artifact_id": "V1PM_T002", "stage": "v1pm", "relative_path": OUT_T_SIM.name, "rows": str(len(t_sim)), "header_present": "true", "role": "similarity_results"},
        {"artifact_id": "V1PM_T003", "stage": "v1pm", "relative_path": OUT_T_PCA.name, "rows": str(len(t_pca)), "header_present": "true", "role": "pca_cluster_results"},
        {"artifact_id": "V1PM_T004", "stage": "v1pm", "relative_path": OUT_T_XW.name, "rows": str(len(t_xw)), "header_present": "true", "role": "protocol_c_crosswalk"},
    ]

    for label, rows in (("v1pm_emb", t_emb), ("v1pm_sim", t_sim), ("v1pm_pca", t_pca),
                        ("v1pm_xw", t_xw), ("v1pm_manifest", manifest), ("v1pm_summary", summary)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)

    write_csv(OUT_T_EMB, t_emb, T_EMB_FIELDS)
    write_csv(OUT_T_SIM, t_sim, T_SIM_FIELDS)
    write_csv(OUT_T_PCA, t_pca, T_PCA_FIELDS)
    write_csv(OUT_T_XW, t_xw, T_XW_FIELDS)
    write_csv(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema(SCHEMA_DIR / "dino_tcc_table_embedding_inventory_v1pm_schema.csv", T_EMB_FIELDS, "v1pm_tcc_embedding_inventory")
    write_schema(SCHEMA_DIR / "dino_tcc_table_similarity_results_v1pm_schema.csv", T_SIM_FIELDS, "v1pm_tcc_similarity_results")
    write_schema(SCHEMA_DIR / "dino_tcc_table_pca_cluster_results_v1pm_schema.csv", T_PCA_FIELDS, "v1pm_tcc_pca_cluster_results")
    write_schema(SCHEMA_DIR / "dino_tcc_table_protocol_c_crosswalk_v1pm_schema.csv", T_XW_FIELDS, "v1pm_tcc_protocol_c_crosswalk")
    write_schema(SCHEMA_DIR / "dino_tcc_results_manifest_v1pm_schema.csv", MANIFEST_FIELDS, "v1pm_tcc_manifest")
    write_schema(SCHEMA_DIR / "dino_tcc_results_scientific_summary_v1pm_schema.csv", SUMMARY_FIELDS, "v1pm_tcc_scientific_summary")

    write_doc(DOC, "v1pm — DINO TCC Results Bundle", [
        "## Objetivo",
        "Consolidar v1pg-v1pl em tabelas TCC-ready, manifest e summary científico. "
        "Reframe para escrita, sem recalcular análises.",
        "## Interpretação metodológica (texto para o TCC)",
        TCC_TEXT,
        "## Papel do DINOv2",
        "DINOv2 with registers é representação visual/semântica review-only — não "
        "validador de evento, não criador de rótulo, não treinador de classificador.",
        f"## Status final",
        f"Status final da camada DINO: **{final}**.",
    ])
    print(f"[v1pm] emb={len(t_emb)} sim={len(t_sim)} pca={len(t_pca)} xw={len(t_xw)} status={final}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pm dino tcc results bundle").parse_args()
    run()
