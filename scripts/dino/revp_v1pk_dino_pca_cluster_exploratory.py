"""REV-P v1pk — DINO PCA / cluster exploratory layer.

Projects VALID review-only embeddings to 2D (PCA) and assigns exploratory
clusters. A cluster is NOT a class: `can_be_used_as_class`, `can_create_label`
and `can_train_model` are always false. With fewer than 2 valid embeddings the
outputs are empty with header (fail-closed).
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from revp_v1pg_v1pm_dino_representation_common import (
    DATASETS, DOCS, SCHEMAS,
    _f, _p, assert_no_forbidden_true, kmeans_simple, load_valid_embeddings,
    pca_2d, require_no_abs_paths, source_root, write_csv, write_doc, write_schema,
)

IN_DISCOVERY = _p("REVP_V1PK_IN_DISCOVERY", DATASETS / "dino_artifact_discovery_v1pg.csv")
IN_REGISTRY = _p("REVP_V1PK_IN_REGISTRY", DATASETS / "dino_embedding_feature_store_registry_v1ph.csv")
OUT_PCA = _p("REVP_V1PK_OUT_PCA", DATASETS / "dino_pca_projection_v1pk.csv")
OUT_CLUSTER = _p("REVP_V1PK_OUT_CLUSTER", DATASETS / "dino_cluster_exploratory_v1pk.csv")
OUT_SUMMARY = _p("REVP_V1PK_OUT_SUMMARY", DATASETS / "dino_pca_cluster_summary_v1pk.csv")
SCHEMA_PCA = _p("REVP_V1PK_SCHEMA_PCA", SCHEMAS / "dino_pca_projection_v1pk_schema.csv")
SCHEMA_CLUSTER = _p("REVP_V1PK_SCHEMA_CLUSTER", SCHEMAS / "dino_cluster_exploratory_v1pk_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1PK_SCHEMA_SUMMARY", SCHEMAS / "dino_pca_cluster_summary_v1pk_schema.csv")
DOC = _p("REVP_V1PK_DOC", DOCS / "revp_v1pk_dino_pca_cluster_exploratory.md")

PCA_FIELDS = [
    "patch_id", "alias", "region", "pca_x", "pca_y",
    "explained_variance_ratio_x", "explained_variance_ratio_y",
    "representation_use", "can_create_label", "can_train_model", "notes",
]
CLUSTER_FIELDS = [
    "patch_id", "alias", "region", "exploratory_cluster_id", "cluster_method",
    "cluster_k", "cluster_use", "can_be_used_as_class", "can_create_label",
    "can_train_model", "notes",
]
SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _choose_k(n: int) -> int:
    if "REVP_V1PK_K" in os.environ:
        return max(1, min(int(os.environ["REVP_V1PK_K"]), n))
    return max(2, min(3, n))


def build(embs: list[dict[str, Any]]) -> tuple[list[dict], list[dict], int, tuple[float, float]]:
    if len(embs) < 2:
        return ([], [], 0, (0.0, 0.0))
    vectors = [e["vector"] for e in embs]
    coords, (evx, evy) = pca_2d(vectors)
    k = _choose_k(len(embs))
    labels = kmeans_simple(vectors, k)

    pca_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for e, (cx, cy), lab in zip(embs, coords, labels):
        pca_rows.append({
            "patch_id": e["patch_id"], "alias": e["alias"], "region": e["region"],
            "pca_x": _f(cx), "pca_y": _f(cy),
            "explained_variance_ratio_x": _f(evx), "explained_variance_ratio_y": _f(evy),
            "representation_use": "EXPLORATORY_REPRESENTATION_ONLY",
            "can_create_label": "false", "can_train_model": "false", "notes": "",
        })
        cluster_rows.append({
            "patch_id": e["patch_id"], "alias": e["alias"], "region": e["region"],
            "exploratory_cluster_id": f"EXPL_C{lab}", "cluster_method": "kmeans_deterministic",
            "cluster_k": str(k), "cluster_use": "EXPLORATORY_REPRESENTATION_ONLY",
            "can_be_used_as_class": "false", "can_create_label": "false",
            "can_train_model": "false", "notes": "cluster_is_not_a_class",
        })
    return (pca_rows, cluster_rows, k, (evx, evy))


def run() -> None:
    embs = load_valid_embeddings(source_root(), IN_DISCOVERY, IN_REGISTRY)
    pca_rows, cluster_rows, k, _ = build(embs)
    for label, rows in (("v1pk_pca", pca_rows), ("v1pk_cluster", cluster_rows)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    n = len(embs)
    n_clusters = len({r["exploratory_cluster_id"] for r in cluster_rows})
    summary = [
        {"stat_key": "valid_embeddings", "stat_value": str(n)},
        {"stat_key": "pca_rows_generated", "stat_value": str(len(pca_rows))},
        {"stat_key": "exploratory_clusters_generated", "stat_value": str(n_clusters)},
        {"stat_key": "cluster_k", "stat_value": str(k)},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "pca_cluster_status",
         "stat_value": "PCA_CLUSTER_READY_REVIEW_ONLY" if n >= 2 else "PCA_CLUSTER_FAIL_CLOSED_LT2"},
    ]

    write_csv(OUT_PCA, pca_rows, PCA_FIELDS)
    write_csv(OUT_CLUSTER, cluster_rows, CLUSTER_FIELDS)
    write_csv(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema(SCHEMA_PCA, PCA_FIELDS, "v1pk_dino_pca_projection")
    write_schema(SCHEMA_CLUSTER, CLUSTER_FIELDS, "v1pk_dino_cluster_exploratory")
    write_schema(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1pk_dino_pca_cluster_summary")
    write_doc(DOC, "v1pk — DINO PCA / Cluster Exploratory", [
        "## Objetivo",
        "Projetar embeddings válidos review-only em 2D (PCA) e atribuir clusters "
        "exploratórios. Com menos de 2 embeddings válidos, saídas vazias com header.",
        "## Cluster não é classe",
        "Os clusters são agrupamentos exploratórios de coerência visual/semântica. "
        "`can_be_used_as_class`, `can_create_label` e `can_train_model` são sempre "
        "false. Nenhum cluster vira classe operacional ou rótulo.",
        f"## Resultado",
        f"Embeddings válidos: {n}. Linhas PCA: {len(pca_rows)}. Clusters: {n_clusters}.",
    ])
    print(f"[v1pk] valid={n} pca={len(pca_rows)} clusters={n_clusters} k={k}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1pk dino pca cluster exploratory").parse_args()
    run()
