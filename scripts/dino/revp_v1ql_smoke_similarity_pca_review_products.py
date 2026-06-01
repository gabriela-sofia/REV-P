"""REV-P v1ql — Smoke similarity / PCA review products.

Generates review-only exploratory products from the consolidated smoke
representation (v1qk): top-k cosine neighbors, a bounded long-form similarity
matrix, a 2D PCA projection, and deterministic exploratory clusters (n>=4).

Explicitly: clusters are NOT classes; similarity does NOT validate an event;
PCA does NOT validate an event. Nothing here becomes a label or target.
Fail-closed (headers only) when fewer than 2 valid vectors are available.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, EXPECTED_DINO_DIM, SCHEMAS,
    _f, _p, assert_no_forbidden_true, cosine_similarity, env_int,
    exploratory_clusters, parse_embedding_from_row, pca_2d_review, read_csv,
    require_no_abs_paths, validate_vector, write_csv, write_doc, write_schema,
)

IN_STORE = _p("REVP_V1QL_IN_STORE", DATASETS / "dino_representation_feature_store_with_smoke_v1qk.csv")
OUT_NEIGH = _p("REVP_V1QL_OUT_NEIGH", DATASETS / "dino_smoke_similarity_neighbors_v1ql.csv")
OUT_MATRIX = _p("REVP_V1QL_OUT_MATRIX", DATASETS / "dino_smoke_similarity_matrix_long_v1ql.csv")
OUT_PCA = _p("REVP_V1QL_OUT_PCA", DATASETS / "dino_smoke_pca_projection_v1ql.csv")
OUT_CLUST = _p("REVP_V1QL_OUT_CLUST", DATASETS / "dino_smoke_exploratory_clusters_v1ql.csv")
OUT_SUM = _p("REVP_V1QL_OUT_SUM", DATASETS / "dino_smoke_review_products_summary_v1ql.csv")
SCH_NEIGH = _p("REVP_V1QL_SCH_NEIGH", SCHEMAS / "dino_smoke_similarity_neighbors_v1ql_schema.csv")
SCH_MATRIX = _p("REVP_V1QL_SCH_MATRIX", SCHEMAS / "dino_smoke_similarity_matrix_long_v1ql_schema.csv")
SCH_PCA = _p("REVP_V1QL_SCH_PCA", SCHEMAS / "dino_smoke_pca_projection_v1ql_schema.csv")
SCH_CLUST = _p("REVP_V1QL_SCH_CLUST", SCHEMAS / "dino_smoke_exploratory_clusters_v1ql_schema.csv")
SCH_SUM = _p("REVP_V1QL_SCH_SUM", SCHEMAS / "dino_smoke_review_products_summary_v1ql_schema.csv")
DOC = _p("REVP_V1QL_DOC", DOCS / "revp_v1ql_smoke_similarity_pca_review_products.md")

NEIGH_FIELDS = [
    "neighbor_id", "source_embedding_id", "source_patch_id", "source_region",
    "neighbor_embedding_id", "neighbor_patch_id", "neighbor_region", "rank",
    "cosine_similarity", "similarity_validates_event", "cluster_is_label",
    "can_create_label", "can_train_model",
]
MATRIX_FIELDS = [
    "pair_id", "embedding_id_a", "patch_id_a", "embedding_id_b", "patch_id_b",
    "cosine_similarity", "similarity_validates_event",
]
PCA_FIELDS = [
    "pca_id", "embedding_id", "patch_id", "region", "pc1", "pc2",
    "explained_variance_pc1", "explained_variance_pc2", "pca_method",
    "pca_validates_event", "cluster_is_label", "can_create_label",
]
CLUST_FIELDS = [
    "cluster_row_id", "embedding_id", "patch_id", "region", "cluster_index",
    "cluster_is_label", "similarity_validates_event", "can_create_label",
    "can_train_model", "target_created",
]
SUM_FIELDS = ["stat_key", "stat_value"]

TOPK = 5
MATRIX_MAX_PAIRS = 2000


def _load_valid(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("vector_valid") == "false":
            continue
        vec = parse_embedding_from_row(r)
        vstatus, _ = validate_vector(vec, EXPECTED_DINO_DIM)
        if vstatus != "VALID_REVIEW_ONLY" or r.get("notes") == "duplicate":
            continue
        assert vec is not None
        out.append({
            "embedding_id": r.get("embedding_id", "") or r.get("representation_id", ""),
            "patch_id": (r.get("patch_id", "") or "UNKNOWN").upper(),
            "region": r.get("region", "UNKNOWN"), "vector": vec,
        })
    return out


def build(rows: list[dict[str, str]]) -> dict[str, Any]:
    items = _load_valid(rows)
    n = len(items)
    neigh: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    pca: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    method = "NONE"

    if n < 2:
        return {"n": n, "neigh": neigh, "matrix": matrix, "pca": pca,
                "clusters": clusters, "method": method,
                "status": "DINO_SMOKE_REVIEW_PRODUCTS_FAIL_CLOSED_N_LT_2"}

    topk = min(env_int("REVP_DINO_SMOKE_TOPK", TOPK), n - 1)
    nid = 0
    pid_pair = 0
    for i, a in enumerate(items):
        sims = []
        for j, b in enumerate(items):
            if i == j:
                continue
            cs = cosine_similarity(a["vector"], b["vector"])
            sims.append((cs, j))
            if i < j and len(matrix) < MATRIX_MAX_PAIRS:
                pid_pair += 1
                matrix.append({
                    "pair_id": f"V1QL_PAIR_{pid_pair:06d}",
                    "embedding_id_a": a["embedding_id"], "patch_id_a": a["patch_id"],
                    "embedding_id_b": b["embedding_id"], "patch_id_b": b["patch_id"],
                    "cosine_similarity": _f(cs), "similarity_validates_event": "false",
                })
        sims.sort(key=lambda t: (-(t[0] if t[0] == t[0] else -1e9)))
        for rank, (cs, j) in enumerate(sims[:topk], 1):
            nid += 1
            b = items[j]
            neigh.append({
                "neighbor_id": f"V1QL_NB_{nid:06d}",
                "source_embedding_id": a["embedding_id"], "source_patch_id": a["patch_id"],
                "source_region": a["region"], "neighbor_embedding_id": b["embedding_id"],
                "neighbor_patch_id": b["patch_id"], "neighbor_region": b["region"],
                "rank": str(rank), "cosine_similarity": _f(cs),
                "similarity_validates_event": "false", "cluster_is_label": "false",
                "can_create_label": "false", "can_train_model": "false",
            })

    vectors = [it["vector"] for it in items]
    coords, (ex, ey), method = pca_2d_review(vectors)
    for i, (it, (x, y)) in enumerate(zip(items, coords), 1):
        pca.append({
            "pca_id": f"V1QL_PCA_{i:05d}", "embedding_id": it["embedding_id"],
            "patch_id": it["patch_id"], "region": it["region"],
            "pc1": _f(x), "pc2": _f(y), "explained_variance_pc1": _f(ex),
            "explained_variance_pc2": _f(ey), "pca_method": method,
            "pca_validates_event": "false", "cluster_is_label": "false",
            "can_create_label": "false",
        })

    cluster_idx = exploratory_clusters(vectors, env_int("REVP_DINO_SMOKE_K", 3))
    if cluster_idx:
        for i, (it, ci) in enumerate(zip(items, cluster_idx), 1):
            clusters.append({
                "cluster_row_id": f"V1QL_CL_{i:05d}", "embedding_id": it["embedding_id"],
                "patch_id": it["patch_id"], "region": it["region"],
                "cluster_index": str(ci), "cluster_is_label": "false",
                "similarity_validates_event": "false", "can_create_label": "false",
                "can_train_model": "false", "target_created": "false",
            })

    return {"n": n, "neigh": neigh, "matrix": matrix, "pca": pca,
            "clusters": clusters, "method": method,
            "status": "DINO_SMOKE_REVIEW_PRODUCTS_READY_REVIEW_ONLY"}


def run() -> None:
    rows = read_csv(IN_STORE)
    res = build(rows)
    for label, data in (("v1ql_neigh", res["neigh"]), ("v1ql_matrix", res["matrix"]),
                        ("v1ql_pca", res["pca"]), ("v1ql_clusters", res["clusters"])):
        require_no_abs_paths(data, label)
        assert_no_forbidden_true(data, label)
    summary = [
        {"stat_key": "valid_vectors", "stat_value": str(res["n"])},
        {"stat_key": "neighbor_rows", "stat_value": str(len(res["neigh"]))},
        {"stat_key": "matrix_pairs", "stat_value": str(len(res["matrix"]))},
        {"stat_key": "pca_rows", "stat_value": str(len(res["pca"]))},
        {"stat_key": "pca_method", "stat_value": res["method"]},
        {"stat_key": "cluster_rows", "stat_value": str(len(res["clusters"]))},
        {"stat_key": "cluster_is_label", "stat_value": "false"},
        {"stat_key": "similarity_validates_event", "stat_value": "false"},
        {"stat_key": "pca_validates_event", "stat_value": "false"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "final_status", "stat_value": res["status"]},
    ]
    require_no_abs_paths(summary, "v1ql_summary")
    assert_no_forbidden_true(summary, "v1ql_summary")
    write_csv(OUT_NEIGH, res["neigh"], NEIGH_FIELDS)
    write_csv(OUT_MATRIX, res["matrix"], MATRIX_FIELDS)
    write_csv(OUT_PCA, res["pca"], PCA_FIELDS)
    write_csv(OUT_CLUST, res["clusters"], CLUST_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_NEIGH, NEIGH_FIELDS, "v1ql_smoke_similarity_neighbors")
    write_schema(SCH_MATRIX, MATRIX_FIELDS, "v1ql_smoke_similarity_matrix_long")
    write_schema(SCH_PCA, PCA_FIELDS, "v1ql_smoke_pca_projection")
    write_schema(SCH_CLUST, CLUST_FIELDS, "v1ql_smoke_exploratory_clusters")
    write_schema(SCH_SUM, SUM_FIELDS, "v1ql_smoke_review_products_summary")
    write_doc(DOC, "v1ql — Smoke Similarity / PCA Review Products", [
        "## Objetivo",
        "Gerar produtos exploratórios review-only a partir da representação smoke "
        "(v1qk): vizinhos cosine top-k, matriz long-form limitada, projeção PCA 2D e "
        "clusters exploratórios determinísticos (n>=4).",
        "## Fronteira metodológica",
        "Clusters não são classe. Similaridade não valida evento. PCA não valida "
        "evento. Nenhum produto vira rótulo ou target. Fail-closed com headers se n<2.",
        "## Status",
        f"**{res['status']}**. Vetores válidos: {res['n']}.",
    ])
    print(f"[v1ql] status={res['status']} n={res['n']} neigh={len(res['neigh'])} pca={len(res['pca'])}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ql smoke similarity/pca review products").parse_args()
    run()
