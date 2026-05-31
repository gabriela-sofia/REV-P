"""REV-P v1ps — Smoke embedding review products.

Generates similarity neighbors, PCA projection, exploratory cluster, and
Protocol C crosswalk from v1pr smoke feature store. If fewer than 2 valid
embeddings exist, all outputs are empty with header (fail-closed).
Does NOT alter v1pj/v1pk/v1pl originals.
Cluster ≠ class. Neighbor ≠ event. PCA ≠ validation. DINO ≠ event validator.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

from revp_v1pn_v1pt_dino_execution_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, require_no_abs_paths, write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import (
    _f, cosine_similarity, euclidean_distance, kmeans_simple,
    normalize_region, pca_2d, parse_embedding_from_text, read_csv,
)

IN_STORE = _p("REVP_V1PS_IN_STORE", DATASETS / "dino_smoke_embedding_feature_store_v1pr.csv")
IN_V1OY = _p("REVP_V1PS_IN_V1OY", DATASETS / "recife_ground_truth_candidate_decision_audit_v1oy.csv")

OUT_NEIGHBORS = _p("REVP_V1PS_OUT_NEIGHBORS", DATASETS / "dino_smoke_similarity_neighbors_v1ps.csv")
OUT_PCA = _p("REVP_V1PS_OUT_PCA", DATASETS / "dino_smoke_pca_projection_v1ps.csv")
OUT_CLUSTER = _p("REVP_V1PS_OUT_CLUSTER", DATASETS / "dino_smoke_cluster_exploratory_v1ps.csv")
OUT_XW = _p("REVP_V1PS_OUT_XW", DATASETS / "dino_smoke_protocol_c_crosswalk_v1ps.csv")
OUT_SUM = _p("REVP_V1PS_OUT_SUM", DATASETS / "dino_smoke_review_products_summary_v1ps.csv")

SCH_NB = _p("REVP_V1PS_SCH_NB", SCHEMAS / "dino_smoke_similarity_neighbors_v1ps_schema.csv")
SCH_PCA = _p("REVP_V1PS_SCH_PCA", SCHEMAS / "dino_smoke_pca_projection_v1ps_schema.csv")
SCH_CL = _p("REVP_V1PS_SCH_CL", SCHEMAS / "dino_smoke_cluster_exploratory_v1ps_schema.csv")
SCH_XW = _p("REVP_V1PS_SCH_XW", SCHEMAS / "dino_smoke_protocol_c_crosswalk_v1ps_schema.csv")
SCH_SUM = _p("REVP_V1PS_SCH_SUM", SCHEMAS / "dino_smoke_review_products_summary_v1ps_schema.csv")
DOC = _p("REVP_V1PS_DOC", DOCS / "revp_v1ps_smoke_embedding_review_products.md")

TOP_K = int(os.environ.get("REVP_V1PS_TOP_K", "5"))

NEIGHBOR_FIELDS = [
    "query_patch_id", "query_region", "neighbor_rank", "neighbor_patch_id",
    "neighbor_region", "cosine_similarity", "euclidean_distance", "same_region",
    "representation_use", "can_infer_same_event", "can_create_label", "can_train_model", "notes",
]
PCA_FIELDS = [
    "patch_id", "alias", "region", "pca_x", "pca_y",
    "explained_variance_ratio_x", "explained_variance_ratio_y",
    "representation_use", "can_create_label", "can_train_model", "notes",
]
CLUSTER_FIELDS = [
    "patch_id", "alias", "region", "exploratory_cluster_id", "cluster_method",
    "cluster_k", "cluster_use", "can_be_used_as_class", "can_create_label", "can_train_model", "notes",
]
XW_FIELDS = [
    "crosswalk_id", "patch_id", "region", "embedding_id", "embedding_status",
    "protocol_c_event_id", "protocol_c_candidate_level",
    "dino_can_validate_event", "dino_can_create_label", "dino_can_train_model", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _load_valid(store: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in store:
        if r.get("embedding_status") != "VALID_REVIEW_ONLY":
            continue
        emb_raw = r.get("embedding", "")
        if not emb_raw:
            continue
        vec = parse_embedding_from_text(emb_raw)
        if vec is None or len(vec) != 768:
            continue
        out.append({**r, "vector": vec})
    return out


def _neighbors(embs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(embs):
        sims = []
        for j, other in enumerate(embs):
            if i == j:
                continue
            cos = cosine_similarity(q["vector"], other["vector"])
            euc = euclidean_distance(q["vector"], other["vector"])
            sims.append((cos, euc, other))
        sims.sort(key=lambda t: (-(t[0] if not math.isnan(t[0]) else -1e9), t[1]))
        for rank, (cos, euc, other) in enumerate(sims[:TOP_K], 1):
            rows.append({
                "query_patch_id": q["patch_id"], "query_region": q["region"],
                "neighbor_rank": str(rank), "neighbor_patch_id": other["patch_id"],
                "neighbor_region": other["region"],
                "cosine_similarity": _f(cos), "euclidean_distance": _f(euc),
                "same_region": str(q["region"] == other["region"]).lower(),
                "representation_use": "EXPLORATORY_SIMILARITY_ONLY",
                "can_infer_same_event": "false", "can_create_label": "false",
                "can_train_model": "false", "notes": "",
            })
    return rows


def _pca(embs: list[dict[str, Any]]) -> tuple[list[dict], list[dict], int]:
    vectors = [e["vector"] for e in embs]
    k = max(2, min(3, len(embs)))
    coords, (evx, evy) = pca_2d(vectors)
    labels = kmeans_simple(vectors, k)
    pca_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for e, (cx, cy), lab in zip(embs, coords, labels):
        pca_rows.append({
            "patch_id": e["patch_id"], "alias": e.get("alias", ""), "region": e["region"],
            "pca_x": _f(cx), "pca_y": _f(cy),
            "explained_variance_ratio_x": _f(evx), "explained_variance_ratio_y": _f(evy),
            "representation_use": "EXPLORATORY_REPRESENTATION_ONLY",
            "can_create_label": "false", "can_train_model": "false", "notes": "",
        })
        cluster_rows.append({
            "patch_id": e["patch_id"], "alias": e.get("alias", ""), "region": e["region"],
            "exploratory_cluster_id": f"SMOKE_C{lab}", "cluster_method": "kmeans_deterministic",
            "cluster_k": str(k), "cluster_use": "EXPLORATORY_REPRESENTATION_ONLY",
            "can_be_used_as_class": "false", "can_create_label": "false",
            "can_train_model": "false", "notes": "cluster_is_not_a_class",
        })
    return pca_rows, cluster_rows, k


def _crosswalk(embs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v1oy_rows = read_csv(IN_V1OY)
    oy_by_patch = {r.get("patch_id", "").strip().upper(): r for r in v1oy_rows}
    xw: list[dict[str, Any]] = []
    for i, e in enumerate(embs, 1):
        pid = e.get("patch_id", "").strip().upper()
        dec = oy_by_patch.get(pid, {})
        xw.append({
            "crosswalk_id": f"V1PS_XW_{i:05d}",
            "patch_id": pid, "region": e.get("region", ""),
            "embedding_id": e.get("embedding_id", ""),
            "embedding_status": e.get("embedding_status", ""),
            "protocol_c_event_id": dec.get("event_id", ""),
            "protocol_c_candidate_level": dec.get("candidate_level", "NOT_IN_PROTOCOL_C"),
            "dino_can_validate_event": "false",
            "dino_can_create_label": "false", "dino_can_train_model": "false",
            "notes": "",
        })
    return xw


def run() -> None:
    store = read_csv(IN_STORE)
    # v1pr stores stats only; vectors live in v1pq results; load from store if embedding field present
    embs = _load_valid(store)
    n = len(embs)

    if n >= 2:
        nb = _neighbors(embs)
        pca_rows, cluster_rows, k = _pca(embs)
        xw = _crosswalk(embs)
    else:
        nb, pca_rows, cluster_rows, xw, k = [], [], [], [], 0

    for label, rows in [("v1ps_nb", nb), ("v1ps_pca", pca_rows),
                        ("v1ps_cl", cluster_rows), ("v1ps_xw", xw)]:
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)

    n_cl = len({r["exploratory_cluster_id"] for r in cluster_rows})
    summary = [
        {"stat_key": "valid_smoke_embeddings", "stat_value": str(n)},
        {"stat_key": "smoke_neighbors", "stat_value": str(len(nb))},
        {"stat_key": "smoke_pca_rows", "stat_value": str(len(pca_rows))},
        {"stat_key": "smoke_clusters", "stat_value": str(n_cl)},
        {"stat_key": "crosswalk_rows", "stat_value": str(len(xw))},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "review_products_status",
         "stat_value": "SMOKE_REVIEW_PRODUCTS_READY" if n >= 2 else "SMOKE_REVIEW_PRODUCTS_FAIL_CLOSED_LT2"},
    ]

    write_csv(OUT_NEIGHBORS, nb, NEIGHBOR_FIELDS)
    write_csv(OUT_PCA, pca_rows, PCA_FIELDS)
    write_csv(OUT_CLUSTER, cluster_rows, CLUSTER_FIELDS)
    write_csv(OUT_XW, xw, XW_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_NB, NEIGHBOR_FIELDS, "v1ps_smoke_similarity_neighbors")
    write_schema(SCH_PCA, PCA_FIELDS, "v1ps_smoke_pca_projection")
    write_schema(SCH_CL, CLUSTER_FIELDS, "v1ps_smoke_cluster_exploratory")
    write_schema(SCH_XW, XW_FIELDS, "v1ps_smoke_protocol_c_crosswalk")
    write_schema(SCH_SUM, SUM_FIELDS, "v1ps_smoke_review_products_summary")
    write_doc(DOC, "v1ps — Smoke Embedding Review Products", [
        "## Objetivo",
        "Gerar vizinhos, PCA, cluster e crosswalk a partir de embeddings smoke (v1pr). "
        "Não altera v1pj/v1pk/v1pl originais. Com <2 embeddings, saídas vazias com header.",
        "## Guardrails",
        "Cluster ≠ classe. Vizinho ≠ evento. PCA ≠ validação. DINO não valida Protocolo C.",
        f"## Resultado",
        f"Embeddings válidos: {n}. Vizinhos: {len(nb)}. PCA: {len(pca_rows)}. Clusters: {n_cl}.",
    ])
    print(f"[v1ps] n={n} neighbors={len(nb)} pca={len(pca_rows)} clusters={n_cl}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ps smoke embedding review products").parse_args()
    run()
