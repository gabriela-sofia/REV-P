"""REV-P v1pj — DINO similarity neighbor graph.

Computes top-k cosine neighbors and a long-form similarity matrix over VALID
review-only embeddings. Similarity is an exploratory visual/semantic coherence
signal — it NEVER infers same-event, creates labels, or trains models. If fewer
than 2 valid embeddings exist, outputs are empty with header (fail-closed).
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, SCHEMAS,
    _f, _p, assert_no_forbidden_true, cosine_similarity, euclidean_distance,
    load_valid_embeddings, require_no_abs_paths, source_root, write_csv, write_doc, write_schema,
)

IN_DISCOVERY = _p("REVP_V1PJ_IN_DISCOVERY", DATASETS / "dino_artifact_discovery_v1pg.csv")
IN_REGISTRY = _p("REVP_V1PJ_IN_REGISTRY", DATASETS / "dino_embedding_feature_store_registry_v1ph.csv")
OUT_NEIGHBORS = _p("REVP_V1PJ_OUT_NEIGHBORS", DATASETS / "dino_similarity_neighbors_v1pj.csv")
OUT_MATRIX = _p("REVP_V1PJ_OUT_MATRIX", DATASETS / "dino_similarity_matrix_long_v1pj.csv")
OUT_SUMMARY = _p("REVP_V1PJ_OUT_SUMMARY", DATASETS / "dino_similarity_summary_v1pj.csv")
SCHEMA_NEIGHBORS = _p("REVP_V1PJ_SCHEMA_NEIGHBORS", SCHEMAS / "dino_similarity_neighbors_v1pj_schema.csv")
SCHEMA_MATRIX = _p("REVP_V1PJ_SCHEMA_MATRIX", SCHEMAS / "dino_similarity_matrix_long_v1pj_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PJ_SCHEMA_SUMMARY", SCHEMAS / "dino_similarity_summary_v1pj_schema.csv")
DOC = _p("REVP_V1PJ_DOC", DOCS / "revp_v1pj_dino_similarity_neighbor_graph.md")

TOP_K = int(os.environ.get("REVP_V1PJ_TOP_K", "5"))

NEIGHBOR_FIELDS = [
    "query_patch_id", "query_region", "neighbor_rank", "neighbor_patch_id",
    "neighbor_region", "cosine_similarity", "euclidean_distance", "same_region",
    "representation_use", "can_infer_same_event", "can_create_label",
    "can_train_model", "notes",
]
MATRIX_FIELDS = [
    "patch_id_a", "patch_id_b", "region_a", "region_b",
    "cosine_similarity", "euclidean_distance", "same_region", "representation_use",
]
SUMMARY_FIELDS = ["stat_key", "stat_value"]


def build_graph(embs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    neighbors: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    if len(embs) < 2:
        return ([], [])
    for i, q in enumerate(embs):
        sims: list[tuple[float, float, dict[str, Any]]] = []
        for j, other in enumerate(embs):
            if i == j:
                continue
            cos = cosine_similarity(q["vector"], other["vector"])
            euc = euclidean_distance(q["vector"], other["vector"])
            sims.append((cos, euc, other))
            if j > i:
                matrix.append({
                    "patch_id_a": q["patch_id"], "patch_id_b": other["patch_id"],
                    "region_a": q["region"], "region_b": other["region"],
                    "cosine_similarity": _f(cos), "euclidean_distance": _f(euc),
                    "same_region": str(q["region"] == other["region"]).lower(),
                    "representation_use": "EXPLORATORY_SIMILARITY_ONLY",
                })
        sims.sort(key=lambda t: (-(t[0] if t[0] == t[0] else -1e9), t[1]))
        for rank, (cos, euc, other) in enumerate(sims[:TOP_K], 1):
            neighbors.append({
                "query_patch_id": q["patch_id"], "query_region": q["region"],
                "neighbor_rank": str(rank), "neighbor_patch_id": other["patch_id"],
                "neighbor_region": other["region"],
                "cosine_similarity": _f(cos), "euclidean_distance": _f(euc),
                "same_region": str(q["region"] == other["region"]).lower(),
                "representation_use": "EXPLORATORY_SIMILARITY_ONLY",
                "can_infer_same_event": "false",
                "can_create_label": "false", "can_train_model": "false",
                "notes": "",
            })
    return (neighbors, matrix)


def run() -> None:
    embs = load_valid_embeddings(source_root(), IN_DISCOVERY, IN_REGISTRY)
    neighbors, matrix = build_graph(embs)
    for label, rows in (("v1pj_neighbors", neighbors), ("v1pj_matrix", matrix)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    n = len(embs)
    summary = [
        {"stat_key": "valid_embeddings", "stat_value": str(n)},
        {"stat_key": "neighbor_pairs_generated", "stat_value": str(len(neighbors))},
        {"stat_key": "matrix_pairs_generated", "stat_value": str(len(matrix))},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "similarity_status",
         "stat_value": "SIMILARITY_GRAPH_READY_REVIEW_ONLY" if n >= 2 else "SIMILARITY_FAIL_CLOSED_LT2"},
    ]

    write_csv(OUT_NEIGHBORS, neighbors, NEIGHBOR_FIELDS)
    write_csv(OUT_MATRIX, matrix, MATRIX_FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema(SCHEMA_NEIGHBORS, NEIGHBOR_FIELDS, "v1pj_dino_similarity_neighbors")
    write_schema(SCHEMA_MATRIX, MATRIX_FIELDS, "v1pj_dino_similarity_matrix_long")
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pj_dino_similarity_summary")
    write_doc(DOC, "v1pj — DINO Similarity Neighbor Graph", [
        "## Objetivo",
        "Calcular vizinhos top-k por similaridade de cosseno e matriz long-form "
        "sobre embeddings válidos review-only. Com menos de 2 embeddings válidos, "
        "saídas vazias com header (fail-closed).",
        "## Interpretação",
        "Similaridade é sinal exploratório de coerência visual/semântica entre patches. "
        "`can_infer_same_event`, `can_create_label` e `can_train_model` são sempre false. "
        "Vizinhança não implica mesmo evento observado.",
        f"## Resultado",
        f"Embeddings válidos: {n}. Pares de vizinhos: {len(neighbors)}.",
    ])
    print(f"[v1pj] valid={n} neighbors={len(neighbors)} matrix={len(matrix)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pj dino similarity neighbor graph").parse_args()
    run()
