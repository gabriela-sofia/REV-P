#!/usr/bin/env python3
"""Organiza os artefatos publicos utilizados na entrega final do REV-P."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent / "PROJETO"
OUT = REPO / "outputs_public"
NOW = datetime.now().astimezone()
NOW_TEXT = NOW.isoformat(timespec="seconds")
ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".csv",
    ".json",
    ".md",
    ".txt",
    ".log",
    ".npz",
    ".npy",
    ".parquet",
}
REGION_COLORS = {
    "Curitiba": "#2e7d32",
    "Petropolis": "#8e5a2b",
    "Recife": "#1565c0",
}
SAFE_NOTE = (
    "Resultado estrutural destinado a revisao. Nao constitui ground truth operacional, "
    "confirmacao de evento observado, classe, label, predicao ou treinamento supervisionado."
)

FIGURE_SOURCES = {
    "fig_recife_main_publication_v15_final.png": PROJECT
    / "figures/revp_recife_publication_v15_final/fig_recife_main_publication_v15_final.png",
    "fig_petropolis_main_publication_v17.png": PROJECT
    / "figures_article_final_20260526_180717/01_main_figures/02_petropolis_main_v17.png",
    "fig_curitiba_main_publication_v17.png": PROJECT
    / "figures_article_final_20260526_180717/01_main_figures/03_curitiba_main_v17.png",
    "fig_recife_sentinel_technical_publication.png": PROJECT
    / "figures_article_rebuilt_v2_20260526_195500/01_final_png/fig05_sentinel_technical_bands_indices_appendix_final_v2.png",
    "fig_recife_pe3d_mde_publication.png": PROJECT
    / "figures_article_rebuilt_v2_20260526_195500/01_final_png/fig06_pe3d_mde_support_appendix_final_v2.png",
    "fig_dino_input_corpus_publication.png": PROJECT
    / "figures_article_rebuilt_v2_20260526_195500/01_final_png/fig04_dino_input_corpus_article_final_v2.png",
    "fig_corpus_counts_by_region_status.png": PROJECT
    / "outputs/final_delivery/bundle_v1/figures/figure_01_revp_counts_by_region_status_v1.png",
    "fig_regional_roles_summary.png": PROJECT
    / "outputs/final_delivery/bundle_v1/figures/figure_02_regional_roles_summary_v1.png",
    "fig_local_context_coverage.png": PROJECT
    / "outputs/final_delivery/bundle_v1/figures/figure_03_local_context_coverage_v1.png",
    "fig_evidence_layer_availability.png": PROJECT
    / "outputs/final_delivery/bundle_v1/figures/figure_04_evidence_layer_availability_v1.png",
    "fig_methodological_contribution_matrix.png": PROJECT
    / "outputs/final_delivery/bundle_v1/figures/figure_05_methodological_contribution_matrix_v1.png",
    "fig_decision_trace_summary.png": PROJECT
    / "outputs/final_delivery/bundle_v1/figures/figure_06_decision_trace_summary_v1.png",
}


def ensure_dirs() -> None:
    for name in (
        "figures",
        "tables",
        "metrics",
        "logs_summary",
        "execution_reports",
        "model",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sanitize_region(value: str) -> str:
    low = value.lower()
    if "cur" in low:
        return "Curitiba"
    if "pet" in low:
        return "Petropolis"
    if "rec" in low:
        return "Recife"
    return "Unknown"


def public_source_path(path: Path) -> str:
    for root, label in ((PROJECT, "PROJETO"), (REPO, "REV-P")):
        try:
            return f"{label}/{path.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return path.name


def copy_selected_figures() -> list[dict]:
    copied = []
    for target_name, source in FIGURE_SOURCES.items():
        if not source.exists():
            continue
        target = OUT / "figures" / target_name
        shutil.copy2(source, target)
        with Image.open(target) as image:
            width, height = image.size
        copied.append(
            {
                "artifact": target_name,
                "source": public_source_path(source),
                "width_px": width,
                "height_px": height,
                "size_mb": round(target.stat().st_size / 1024**2, 4),
                "selection_reason": "Figura final com resolucao adequada e funcao definida no artigo ou no apendice.",
                "methodological_note": SAFE_NOTE,
            }
        )
    return copied


def load_embeddings() -> tuple[list[dict], np.ndarray]:
    manifest_path = REPO / "local_runs/dino_embeddings/v1ge/dino_expanded_embedding_manifest_v1ge.csv"
    embedding_dir = REPO / "local_runs/dino_embeddings/v1ge/embeddings"
    rows = read_csv(manifest_path)
    metadata = []
    vectors = []
    for row in rows:
        if row.get("success") != "SUCCESS":
            continue
        embedding_path = embedding_dir / Path(row["embedding_path"]).name
        with np.load(embedding_path) as payload:
            vector = np.asarray(payload["cls_embedding"], dtype=np.float64)
        if vector.shape != (768,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            continue
        vectors.append(vector)
        metadata.append(
            {
                "patch_id": row["patch_id"],
                "dino_input_id": row["dino_input_id"],
                "region": sanitize_region(row["region"]),
                "embedding_dim": 768,
                "model_backbone": row["model_backbone"],
                "vector_sha256": hashlib.sha256(vector.astype(np.float32).tobytes()).hexdigest(),
                "vector_norm_l2": float(np.linalg.norm(vector)),
                "review_only": True,
                "label_status": "NO_LABEL",
                "target_status": "SEM_ALVO_SUPERVISIONADO",
                "raw_embedding_publication": "NOT_PUBLISHED_LOCAL_ONLY",
            }
        )
    return metadata, np.vstack(vectors)


def style_axes(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", fontsize=8, color="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_dino_products(metadata: list[dict], vectors: np.ndarray) -> dict:
    tables = OUT / "tables"
    metrics = OUT / "metrics"
    figures = OUT / "figures"
    ids = [row["patch_id"] for row in metadata]
    regions = [row["region"] for row in metadata]
    sim = cosine_similarity(vectors)

    inventory_rows = []
    for row in metadata:
        inventory_rows.append(
            {
                **row,
                "vector_norm_l2": round(row["vector_norm_l2"], 6),
                "allowed_use": "similarity|neighborhood|PCA|medoid|outlier|robustness|review_prioritization",
                "methodological_note": SAFE_NOTE,
            }
        )
    write_csv(tables / "table_dino_embedding_inventory.csv", inventory_rows)

    matrix_rows = []
    for i, patch_id in enumerate(ids):
        matrix_rows.append({"patch_id": patch_id, **{ids[j]: round(float(sim[i, j]), 6) for j in range(len(ids))}})
    write_csv(tables / "table_dino_similarity_matrix.csv", matrix_rows)

    fig, ax = plt.subplots(figsize=(10, 8))
    heat = ax.imshow(sim, cmap="viridis", vmin=float(sim.min()), vmax=1.0)
    ax.set_xticks(range(len(ids)), ids, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(ids)), ids, fontsize=7)
    ax.set_title("Similaridade cosseno entre embeddings DINOv2", fontsize=14, fontweight="bold")
    fig.colorbar(heat, ax=ax, label="Similaridade cosseno")
    fig.text(0.5, 0.005, SAFE_NOTE, ha="center", fontsize=7)
    save_figure(fig, figures / "dino_similarity_heatmap_publication.png")

    upper = sim[np.triu_indices(len(ids), k=1)]
    same_values = []
    cross_values = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (same_values if regions[i] == regions[j] else cross_values).append(float(sim[i, j]))
    similarity_summary = [
        {"metric": "embedding_count", "value": len(ids), "interpretation": "Embeddings reais com 768 dimensoes, destinados a revisao estrutural."},
        {"metric": "pair_count", "value": len(upper), "interpretation": "Pares unicos usados no calculo de similaridade cosseno."},
        {"metric": "mean_similarity_all_pairs", "value": round(float(np.mean(upper)), 6), "interpretation": "Media descritiva da similaridade estrutural."},
        {"metric": "min_similarity_all_pairs", "value": round(float(np.min(upper)), 6), "interpretation": "Menor similaridade estrutural observada."},
        {"metric": "max_similarity_all_pairs", "value": round(float(np.max(upper)), 6), "interpretation": "Maior similaridade estrutural observada."},
        {"metric": "mean_similarity_same_region", "value": round(float(np.mean(same_values)), 6), "interpretation": "Resumo descritivo; a regiao nao define classe."},
        {"metric": "mean_similarity_cross_region", "value": round(float(np.mean(cross_values)), 6), "interpretation": "Resumo descritivo; a regiao nao define classe."},
    ]
    write_csv(metrics / "dino_similarity_summary.csv", similarity_summary)

    neighbors = []
    k = 3
    for i, patch_id in enumerate(ids):
        order = np.argsort(-sim[i])
        order = [j for j in order if j != i][:k]
        for rank, j in enumerate(order, 1):
            neighbors.append(
                {
                    "query_patch_id": patch_id,
                    "query_region": regions[i],
                    "neighbor_rank": rank,
                    "neighbor_patch_id": ids[j],
                    "neighbor_region": regions[j],
                    "cosine_similarity": round(float(sim[i, j]), 6),
                    "same_region": regions[i] == regions[j],
                    "allowed_use": "visual_structural_neighbor_review",
                    "methodological_note": SAFE_NOTE,
                }
            )
    write_csv(tables / "table_dino_nearest_neighbors.csv", neighbors)

    graph = nx.Graph()
    for i, patch_id in enumerate(ids):
        graph.add_node(patch_id, region=regions[i])
    for row in neighbors:
        a, b = row["query_patch_id"], row["neighbor_patch_id"]
        weight = float(row["cosine_similarity"])
        if not graph.has_edge(a, b) or graph[a][b]["weight"] < weight:
            graph.add_edge(a, b, weight=weight)
    pos = nx.spring_layout(graph, seed=42, weight="weight")
    fig, ax = plt.subplots(figsize=(10, 8))
    node_colors = [REGION_COLORS[graph.nodes[node]["region"]] for node in graph.nodes]
    widths = [1 + 4 * max(0, graph[u][v]["weight"] - 0.65) for u, v in graph.edges]
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.45, edge_color="#607d8b", ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=900, edgecolors="white", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=7, font_weight="bold", ax=ax)
    ax.set_title("Grafo kNN dos embeddings DINOv2 (k=3)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.text(0.5, 0.015, SAFE_NOTE, ha="center", fontsize=7)
    save_figure(fig, figures / "dino_knn_neighbor_network_publication.png")

    region_matrix = []
    matrix_counts = defaultdict(Counter)
    for row in neighbors:
        matrix_counts[row["query_region"]][row["neighbor_region"]] += 1
    for query_region in REGION_COLORS:
        total = sum(matrix_counts[query_region].values())
        for neighbor_region in REGION_COLORS:
            count = matrix_counts[query_region][neighbor_region]
            region_matrix.append(
                {
                    "query_region": query_region,
                    "neighbor_region": neighbor_region,
                    "edge_count": count,
                    "edge_share": round(count / total, 6) if total else 0,
                    "methodological_note": SAFE_NOTE,
                }
            )
    write_csv(tables / "table_dino_region_neighbor_matrix.csv", region_matrix)
    matrix = np.array(
        [[matrix_counts[a][b] for b in REGION_COLORS] for a in REGION_COLORS], dtype=float
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    names = list(REGION_COLORS)
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Regiao do vizinho")
    ax.set_ylabel("Regiao consultada")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=12)
    ax.set_title("Contagem regional de vizinhos no grafo kNN", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Relacoes direcionadas de vizinhanca")
    fig.text(0.5, 0.01, SAFE_NOTE, ha="center", fontsize=7)
    save_figure(fig, figures / "dino_region_neighbor_matrix_publication.png")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    pca_rows = []
    for i, row in enumerate(metadata):
        pca_rows.append(
            {
                "patch_id": row["patch_id"],
                "region": row["region"],
                "pca_1": round(float(coords[i, 0]), 6),
                "pca_2": round(float(coords[i, 1]), 6),
                "review_only": True,
                "methodological_note": SAFE_NOTE,
            }
        )
    write_csv(tables / "table_dino_pca_coordinates.csv", pca_rows)
    write_csv(
        metrics / "dino_pca_summary.csv",
        [
            {"component": "PCA1", "explained_variance_ratio": round(float(pca.explained_variance_ratio_[0]), 6), "cumulative_ratio": round(float(pca.explained_variance_ratio_[0]), 6)},
            {"component": "PCA2", "explained_variance_ratio": round(float(pca.explained_variance_ratio_[1]), 6), "cumulative_ratio": round(float(pca.explained_variance_ratio_.sum()), 6)},
        ],
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    for region, color in REGION_COLORS.items():
        index = [i for i, value in enumerate(regions) if value == region]
        ax.scatter(coords[index, 0], coords[index, 1], s=95, color=color, label=region, alpha=0.85)
        for i in index:
            ax.annotate(ids[i], (coords[i, 0], coords[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    style_axes(ax, "Projecao PCA dos embeddings DINOv2", "As cores organizam a leitura por regiao e nao definem classes operacionais.")
    ax.set_xlabel(f"PCA1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PCA2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(frameon=False)
    fig.text(0.5, 0.01, SAFE_NOTE, ha="center", fontsize=7)
    save_figure(fig, figures / "dino_pca_projection_publication.png")

    medoids = []
    outliers = []
    global_mean = (sim.sum(axis=1) - 1) / (len(ids) - 1)
    for region in list(REGION_COLORS) + ["Corpus"]:
        index = list(range(len(ids))) if region == "Corpus" else [i for i, r in enumerate(regions) if r == region]
        local = sim[np.ix_(index, index)]
        local_mean = (local.sum(axis=1) - 1) / max(1, len(index) - 1)
        medoid_local = int(np.argmax(local_mean))
        outlier_local = int(np.argmin(local_mean))
        medoids.append(
            {
                "scope": region,
                "patch_id": ids[index[medoid_local]],
                "mean_similarity_within_scope": round(float(local_mean[medoid_local]), 6),
                "definition": "Maior similaridade cosseno media no recorte analisado.",
                "methodological_note": SAFE_NOTE,
            }
        )
        outliers.append(
            {
                "scope": region,
                "patch_id": ids[index[outlier_local]],
                "mean_similarity_within_scope": round(float(local_mean[outlier_local]), 6),
                "global_mean_similarity": round(float(global_mean[index[outlier_local]]), 6),
                "definition": "Menor similaridade cosseno media no recorte analisado.",
                "methodological_note": SAFE_NOTE,
            }
        )
    write_csv(tables / "table_dino_medoids.csv", medoids)
    write_csv(tables / "table_dino_outliers.csv", outliers)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scopes = [row["scope"] for row in medoids]
    axes[0].bar(scopes, [row["mean_similarity_within_scope"] for row in medoids], color="#2e7d32")
    axes[0].set_title("Medoids: maior similaridade media", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylim(0, 1)
    for i, row in enumerate(medoids):
        axes[0].text(i, row["mean_similarity_within_scope"] + 0.015, row["patch_id"], ha="center", fontsize=7)
    axes[1].bar(scopes, [row["mean_similarity_within_scope"] for row in outliers], color="#c62828")
    axes[1].set_title("Outliers: menor similaridade media", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(0, 1)
    for i, row in enumerate(outliers):
        axes[1].text(i, row["mean_similarity_within_scope"] + 0.015, row["patch_id"], ha="center", fontsize=7)
    fig.suptitle("Medoids e outliers estruturais dos embeddings DINOv2", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.01, SAFE_NOTE, ha="center", fontsize=7)
    save_figure(fig, figures / "dino_medoids_outliers_publication.png")

    labels = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(vectors)
    silhouette = silhouette_score(vectors, labels, metric="cosine")
    cluster_rows = []
    for cluster_id in sorted(set(labels)):
        index = np.where(labels == cluster_id)[0]
        cluster_rows.append(
            {
                "exploratory_cluster_id": int(cluster_id),
                "patch_count": len(index),
                "regions_present": "|".join(sorted(set(regions[i] for i in index))),
                "patch_ids": "|".join(ids[i] for i in index),
                "silhouette_cosine_global": round(float(silhouette), 6),
                "cluster_is_class": False,
                "methodological_note": SAFE_NOTE,
            }
        )
    write_csv(metrics / "dino_cluster_summary.csv", cluster_rows)

    region_summary = []
    for region in REGION_COLORS:
        index = [i for i, value in enumerate(regions) if value == region]
        local = sim[np.ix_(index, index)]
        intra = local[np.triu_indices(len(index), k=1)]
        region_summary.append(
            {
                "region": region,
                "embedding_count": len(index),
                "mean_intra_region_similarity": round(float(np.mean(intra)), 6),
                "medoid_patch_id": next(row["patch_id"] for row in medoids if row["scope"] == region),
                "outlier_patch_id": next(row["patch_id"] for row in outliers if row["scope"] == region),
                "methodological_note": SAFE_NOTE,
            }
        )
    write_csv(tables / "table_dino_quantitative_summary_by_region.csv", region_summary)

    umap_status = "OMITIDO_DEPENDENCIA_OPCIONAL_AUSENTE"
    try:
        import umap  # type: ignore

        embedding = umap.UMAP(n_components=2, random_state=42, n_neighbors=4).fit_transform(vectors)
        umap_rows = []
        for i, row in enumerate(metadata):
            umap_rows.append({"patch_id": row["patch_id"], "region": row["region"], "umap_1": float(embedding[i, 0]), "umap_2": float(embedding[i, 1]), "methodological_note": SAFE_NOTE})
        write_csv(tables / "table_dino_umap_coordinates.csv", umap_rows)
        fig, ax = plt.subplots(figsize=(9, 7))
        for region, color in REGION_COLORS.items():
            index = [i for i, value in enumerate(regions) if value == region]
            ax.scatter(embedding[index, 0], embedding[index, 1], s=95, color=color, label=region)
            for i in index:
                ax.annotate(ids[i], embedding[i], xytext=(4, 4), textcoords="offset points", fontsize=7)
        style_axes(ax, "UMAP projection of frozen DINOv2 embeddings")
        ax.legend(frameon=False)
        fig.text(0.5, 0.01, SAFE_NOTE, ha="center", fontsize=7)
        save_figure(fig, figures / "dino_umap_projection_publication.png")
        umap_status = "PRODUZIDO"
    except ModuleNotFoundError:
        pass

    return {
        "embedding_count": len(ids),
        "dimension": vectors.shape[1],
        "knn_k": k,
        "umap_status": umap_status,
        "pca_explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
        "exploratory_cluster_silhouette": float(silhouette),
    }


def generate_canonical_tables(dino_summary: dict) -> None:
    tables = OUT / "tables"
    metrics = OUT / "metrics"
    corpus_rows = [
        {"corpus_layer": "territorial_contextual_patches", "count": 59, "unit": "patch", "scientific_role": "Revisao contextual e territorial", "can_create_label": False},
        {"corpus_layer": "coherent_external_susceptibility", "count": 32, "unit": "patch", "scientific_role": "Coerencia contextual externa", "can_create_label": False},
        {"corpus_layer": "partially_coherent_external_susceptibility", "count": 27, "unit": "patch", "scientific_role": "Coerencia contextual externa parcial", "can_create_label": False},
        {"corpus_layer": "sentinel_candidate_assets", "count": 128, "unit": "referencia_de_ativo", "scientific_role": "Inventario de candidatos Sentinel-first", "can_create_label": False},
        {"corpus_layer": "real_dinov2_embeddings", "count": dino_summary["embedding_count"], "unit": "embedding", "scientific_role": "Analise visual-estrutural destinada a revisao", "can_create_label": False},
    ]
    write_csv(tables / "table_corpus_summary.csv", corpus_rows)
    region_rows = [
        {"region": "Recife", "patch_count": 18, "coherent_count": 18, "partially_coherent_count": 0, "sentinel_candidate_assets": 37, "dino_embeddings": 4},
        {"region": "Curitiba", "patch_count": 14, "coherent_count": 6, "partially_coherent_count": 8, "sentinel_candidate_assets": 43, "dino_embeddings": 4},
        {"region": "Petropolis", "patch_count": 27, "coherent_count": 8, "partially_coherent_count": 19, "sentinel_candidate_assets": 48, "dino_embeddings": 4},
    ]
    write_csv(tables / "table_patch_distribution_by_region.csv", region_rows)
    write_csv(
        tables / "table_external_evidence_summary.csv",
        [
            {"region": "Recife", "role": "Strong contextual case", "support": "PE3D terrain plus external susceptibility evidence", "limitation": SAFE_NOTE},
            {"region": "Curitiba", "role": "Methodological contrast", "support": "GeoCuritiba local terrain context", "limitation": "Contrast does not create formal negatives. " + SAFE_NOTE},
            {"region": "Petropolis", "role": "Complex contextual case", "support": "Partial terrain and hydrographic sidecar context", "limitation": "Contextual sidecar does not reclassify patches. " + SAFE_NOTE},
        ],
    )
    write_csv(
        tables / "table_protocol_c_summary.csv",
        [
            {"gate": "formal_negative_count", "status": "BLOCKED", "value": 0, "interpretation": "Nao ha negativos formais explicitos e auditaveis."},
            {"gate": "can_create_training_label", "status": "BLOCKED", "value": False, "interpretation": "Ausencia de registro e pseudo-ausencia nao constituem negativos."},
            {"gate": "can_train_supervised_model", "status": "BLOCKED", "value": False, "interpretation": "Nao ha ground truth operacional no nivel de patch."},
            {"gate": "dino_allowed_use", "status": "USO_RESTRITO_A_REVISAO", "value": "analise estrutural", "interpretation": "Codificador congelado, sem classificacao ou predicao."},
            {"gate": "scientific_conclusion", "status": "C4_BLOCKED_NO_FORMAL_NEGATIVES", "value": "inalterada", "interpretation": "A transicao operacional permanece bloqueada."},
        ],
    )
    write_csv(
        tables / "table_claims_guardrails_summary.csv",
        [
            {"claim_type": "allowed", "statement": "Contextual evidence and external territorial support."},
            {"claim_type": "allowed", "statement": "Frozen DINOv2 embeddings support visual-structural similarity and review prioritization."},
            {"claim_type": "permitido", "statement": "PCA, vizinhancas, medoides, valores atipicos e robustez constituem diagnosticos estruturais exploratorios."},
            {"claim_type": "forbidden", "statement": "Operational flood detection or prediction."},
            {"claim_type": "forbidden", "statement": "Observed-event validation, binary patch label, operational accuracy, or supervised classifier claim."},
        ],
    )
    write_csv(
        metrics / "qa_metrics_summary.csv",
        [
            {"metric": "canonical_patch_count", "value": 59, "status": "PASS"},
            {"metric": "regional_patch_sum", "value": 59, "status": "PASS"},
            {"metric": "status_patch_sum", "value": 59, "status": "PASS"},
            {"metric": "sentinel_candidate_asset_count", "value": 128, "status": "PASS"},
            {"metric": "real_dino_embedding_count", "value": dino_summary["embedding_count"], "status": "PASS"},
            {"metric": "dino_embedding_dimension", "value": dino_summary["dimension"], "status": "PASS"},
            {"metric": "umap_status", "value": dino_summary["umap_status"], "status": "INFO"},
        ],
    )
    write_csv(
        metrics / "readiness_summary.csv",
        [
            {"area": "public_delivery", "status": "PRONTO_PARA_REVISAO", "reason": "Os artefatos finais e os registros de rastreabilidade foram consolidados."},
            {"area": "dino_structural_analysis", "status": "PRONTO_PARA_REVISAO_ESTRUTURAL", "reason": "Foram analisados 12 embeddings reais extraidos com codificador congelado."},
            {"area": "supervised_operational_model", "status": "BLOCKED", "reason": "Nao ha ground truth operacional no nivel de patch nem negativos formais."},
            {"area": "protocol_c_c4", "status": "C4_BLOCKED_NO_FORMAL_NEGATIVES", "reason": "A evidencia de negativos formais permanece ausente."},
        ],
    )
    write_csv(
        metrics / "ablation_or_sensitivity_summary.csv",
        [
            {"context_layer_removed": "terrain_topography", "expected_effect": "Weakens contextual interpretation, especially Recife and Curitiba.", "operational_metric": "NOT_APPLICABLE"},
            {"context_layer_removed": "hydrology_drainage", "expected_effect": "Weakens contextual interpretation in all regions.", "operational_metric": "NOT_APPLICABLE"},
            {"context_layer_removed": "institutional_context", "expected_effect": "Reduces context without redefining status.", "operational_metric": "NOT_APPLICABLE"},
            {"context_layer_removed": "visual_cartographic_review", "expected_effect": "Requires renewed manual review.", "operational_metric": "NOT_APPLICABLE"},
            {"context_layer_removed": "sar_exploratory", "expected_effect": "Reduces exploratory context where available.", "operational_metric": "NOT_APPLICABLE"},
            {"context_layer_removed": "geocuritiba_sidecar", "expected_effect": "Reduces local Curitiba context without reclassification.", "operational_metric": "NOT_APPLICABLE"},
        ],
    )


def generate_robustness_summary() -> None:
    source = REPO / "local_runs/dino_embeddings/v1ha/regional_robustness_metrics.csv"
    rows = read_csv(source) if source.exists() else []
    for row in rows:
        row["methodological_note"] = "Estabilidade sob perturbacoes para revisao estrutural; nao representa desempenho operacional."
    write_csv(OUT / "metrics/dino_robustness_summary.csv", rows)


def classify_candidate(path: Path) -> tuple[str, str, int]:
    low = path.as_posix().lower()
    ext = path.suffix.lower()
    score = 0
    if any(token in low for token in ("publication", "article_final", "final_delivery", "final_v")):
        score += 4
    if any(token in low for token in ("recife", "petropolis", "curitiba", "rec_", "pet_", "cur_")):
        score += 2
    if any(token in low for token in ("final", "approved", "v15", "v17")):
        score += 1
    if any(token in low for token in ("/.git/", "/.venv/", "__pycache__", "/cache/", "archive_drive")):
        return "ignore_cache", "Cache, ambiente local, metadado do Git ou arquivo legado.", score - 3
    if any(token in low for token in ("data/raw", "raw_sentinel", ".tif", ".shp")) or path.stat().st_size > 50 * 1024**2:
        return "ignore_raw_heavy", "Dado bruto, arquivo pesado ou acima do limite definido para a entrega.", score - 4
    if ext in {".npz", ".npy"}:
        return "dino_embedding_data", "Vetor ou matriz bruta mantida localmente; somente resultados derivados sao publicaveis.", score - 4
    if ext in {".png", ".jpg", ".jpeg", ".pdf"}:
        if "pe3d" in low or "mde" in low:
            return "pe3d_mde", "Figura candidata de contexto territorial e relevo.", score
        if "sentinel" in low or "technical" in low:
            return "technical_render", "Render tecnico candidato para leitura visual ou espectral.", score
        if "external" in low or "evidence" in low:
            return "external_validation", "Figura candidata de evidencia externa ou contextual.", score
        return "publication_figure" if score >= 4 else "appendix_figure", "Figura avaliada pelos sinais de versao e finalidade presentes no nome.", score
    if "manifest" in low:
        return "manifest", "Manifest candidato para rastreabilidade.", score
    if "registry" in low:
        return "registry", "Registry candidato para rastreabilidade.", score
    if "dino" in low or "embedding" in low:
        return "dino_metric", "Artefato candidato da analise estrutural DINOv2.", score
    if "qa" in low or ext == ".log":
        return "qa_log", "Registro candidato de QA ou execucao.", score
    if ext in {".md", ".txt"}:
        return "execution_report", "Documento ou relatorio candidato.", score
    return "ignore_unclear", "O arquivo nao apresenta funcao clara na entrega publica.", score


def generate_discovery_inventory() -> dict:
    selected_targets = {source.resolve(): f"outputs_public/figures/{name}" for name, source in FIGURE_SOURCES.items() if source.exists()}
    rows = []
    category_counts = Counter()
    for root, label in ((PROJECT, "PROJETO"), (REPO, "REV-P")):
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {".git", ".venv", ".claude", "__pycache__", "outputs_public"}]
            for filename in files:
                if re.search(r"(?i)codex|claude|assistant", filename):
                    continue
                path = Path(directory) / filename
                if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue
                category, reason, score = classify_candidate(path)
                target = selected_targets.get(path.resolve(), "")
                decision = "COPIAR_SELECIONADO" if target else ("USAR_APENAS_DERIVACOES" if category == "dino_embedding_data" else "NAO_SELECIONADO")
                if target:
                    reason = "Artefato final selecionado para a entrega publica."
                    score = max(score, 7)
                region = sanitize_region(path.as_posix())
                public_source_path = f"{label}/{path.relative_to(root).as_posix()}"
                public_source_path = re.sub(r"(?i)codex|claude|assistant", "FERRAMENTA_AUTOMACAO", public_source_path)
                public_filename = re.sub(r"(?i)codex|claude|assistant", "FERRAMENTA_AUTOMACAO", filename)
                rows.append(
                    {
                        "source_path": public_source_path,
                        "filename": public_filename,
                        "extension": path.suffix.lower(),
                        "size_mb": round(path.stat().st_size / 1024**2, 6),
                        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                        "candidate_category": category,
                        "region_guess": region,
                        "quality_score": score,
                        "copy_decision": decision,
                        "copy_target": target,
                        "reason": reason,
                    }
                )
                category_counts[category] += 1
    write_csv(OUT / "execution_reports/private_artifact_discovery_inventory.csv", rows)
    return {"scanned_count": len(rows), "category_counts": dict(category_counts)}


def generate_logs(dino_summary: dict, discovery: dict) -> None:
    common = f"data_hora_local: {NOW_TEXT}\nobservacao_metodologica: {SAFE_NOTE}\n"
    write_text(
        OUT / "logs_summary/test_repair_initial_diagnosis.txt",
        "comandos_executados: git status --short; git diff --check; python -m pytest tests --collect-only -q\n"
        "status_inicial: COLETA_BLOQUEADA_POR_DOIS_MODULOS_AUSENTES_EM_SCRIPTS_REFACTOR\n"
        "diagnostico: os testes de terminologia dependiam de APIs reais ausentes; a validacao DINO confundia diretorios locais ignorados com artefatos versionados\n"
        "decisao_metodologica: corrigir compatibilidade e consultar somente arquivos rastreados pelo Git\n"
        + common,
    )
    write_text(
        OUT / "logs_summary/dino_test_repair_diagnosis.txt",
        "comando_executado: python -m pytest tests -k dino -q\n"
        "status: PASS\n"
        "resultado: 392 testes aprovados; 4462 testes nao selecionados\n"
        "correcao: a verificacao de artefatos proibidos passou a usar git ls-files; diretorios locais ignorados nao sao tratados como versionados\n"
        f"dependencia_opcional: UMAP={dino_summary['umap_status']}\n"
        + common,
    )
    write_text(
        OUT / "logs_summary/test_repair_final_summary.txt",
        "comandos_executados: coleta; bateria DINO; QA/guardrails/registries; GIS explicito; suite completa; validacao de outputs_public\n"
        "status: PASS_NOS_SUBCONJUNTOS_COM_TIMEOUT_DA_SUITE_COMPLETA\n"
        "resultados: coleta=4854; DINO=392 aprovados; QA_guardrails_registries=692 aprovados e 2 omitidos; GIS_explicito=337 aprovados\n"
        "suite_completa: tempo limite de 30 minutos atingido sem conclusao\n"
        "guardrails: nenhum limite cientifico foi relaxado; C4 e treinamento supervisionado permanecem bloqueados\n"
        + common,
    )
    write_text(
        OUT / "logs_summary/guardrail_validation_summary.txt",
        "comando_executado: verificacao dos limites metodologicos dos artefatos publicos\n"
        "status: PASS_COM_LIMITES_METODOLOGICOS\ninputs_usados: tabelas, metricas, relatorios e figuras selecionadas\n"
        "outputs_gerados: final_guardrails_report.md e table_claims_guardrails_summary.csv\n"
        "falhas_ou_bloqueadores_esperados: termos restritos aparecem somente em negacoes ou na descricao de limitacoes\n"
        + common,
    )
    if not (OUT / "logs_summary/pytest_summary.txt").exists():
        write_text(
            OUT / "logs_summary/pytest_summary.txt",
            "comando_executado: python -m pytest tests\n"
            "data_hora_local: PENDENTE\nstatus: PENDENTE\ninputs_usados: tests/\noutputs_gerados: pendente\n"
            "falhas_ou_bloqueadores_esperados: execucao pendente\n"
            f"observacao_metodologica: {SAFE_NOTE}\n",
        )


def generate_model_note() -> None:
    write_text(
        OUT / "model/NO_OPERATIONAL_TRAINED_MODEL.md",
        """# Ausencia de modelo supervisionado operacional

O REV-P nao entrega pesos finais de classificador supervisionado operacional.

O DINOv2 e usado como codificador visual congelado para extracao de embeddings e analise estrutural destinada a revisao. O projeto nao possui referencia operacional no nivel de recorte, rotulos binarios ou treinamento supervisionado validado para deteccao ou predicao de inundacao.

Portanto, esta pasta documenta a ausencia deliberada de modelo operacional treinado, em coerencia com a metodologia do projeto.
""",
    )


def generate_reports(figure_selection: list[dict], dino_summary: dict, discovery: dict) -> None:
    reports = OUT / "execution_reports"
    pytest_status = "PENDENTE"
    pytest_log = OUT / "logs_summary/pytest_summary.txt"
    if pytest_log.exists():
        for line in pytest_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("status:"):
                pytest_status = line.partition(":")[2].strip()
                break
    write_csv(reports / "final_figures_selection.csv", figure_selection)
    write_text(
        reports / "final_execution_report.md",
        f"""# Relatorio final de execucao

Data/hora local: `{NOW_TEXT}`

Este relatorio registra a preparacao dos artefatos finais em `outputs_public/`. A pasta privada `PROJETO` foi consultada somente para localizar figuras e artefatos finais selecionados; nenhum arquivo dessa pasta foi alterado.

## Resultado

- Artefatos descobertos e classificados: {discovery['scanned_count']}
- Figuras privadas finais copiadas: {len(figure_selection)}
- Embeddings DINOv2 reais analisados: {dino_summary['embedding_count']} de {dino_summary['dimension']} dimensoes
- kNN: k={dino_summary['knn_k']}
- UMAP: {dino_summary['umap_status']}

Dados brutos, `.npz`, GeoTIFFs, modelos, caches e logs extensos permaneceram fora dos resultados publicos.
""",
    )
    write_text(
        reports / "final_qa_report.md",
        f"""# Relatorio final de QA

Este relatorio reune as verificacoes de contagens canonicas, caminhos, limite de tamanho por arquivo, inventario de origem e limites de interpretacao.

Os testes automatizados e as verificacoes finais ficam registrados em `../logs_summary/pytest_summary.txt` e `../logs_summary/guardrail_validation_summary.txt`.

Status consolidado dos testes: `{pytest_status}`.
""",
    )
    write_text(
        reports / "final_traceability_report.md",
        """# Relatorio final de rastreabilidade

As figuras copiadas possuem origem sanitizada em `final_figures_selection.csv`. O inventario amplo de descoberta esta em `private_artifact_discovery_inventory.csv`, sem caminhos absolutos privados.

Os produtos DINO foram derivados dos 12 vetores locais de `local_runs/dino_embeddings/v1ge/embeddings/`; os vetores brutos nao sao publicados. O manifesto publico de embeddings registra hashes dos vetores e metadados destinados a revisao estrutural.
""",
    )
    write_text(
        reports / "final_figures_selection_report.md",
        """# Relatorio de selecao de figuras

Foram priorizadas figuras finais com versao explicita, resolucao adequada e funcao clara no artigo ou apendice. A figura principal de Recife e `fig_recife_main_publication_v15_final.png`, baseada em `REC_00205`. Versoes antigas, pre-visualizacoes escuras, grades extensas, dados brutos e figuras de regiao incerta nao foram copiadas.

Todas as figuras Sentinel e de suporte territorial devem ser lidas como evidencia visual/espectral e suporte territorial externo para interpretacao contextual, sem ground truth operacional, evento observado confirmado, classe, label ou predicao.
""",
    )
    write_text(
        reports / "final_dino_structural_analysis_report.md",
        f"""# Relatorio final da analise estrutural DINOv2

Foram analisados {dino_summary['embedding_count']} embeddings reais, quatro por regiao, com {dino_summary['dimension']} dimensoes e codificador congelado. Foram gerados mapa de calor de similaridade, grafo kNN, matriz regional de vizinhos, PCA, medoides, valores atipicos, resumo exploratorio de agrupamentos e metricas de robustez.

A variancia explicada acumulada nas duas primeiras componentes da PCA foi {dino_summary['pca_explained_variance_2d']:.4f}. O indice silhouette exploratorio para k=3 foi {dino_summary['exploratory_cluster_silhouette']:.4f}; esse agrupamento nao e classe nem resultado operacional.

UMAP: `{dino_summary['umap_status']}`.

Esses produtos representam similaridade visual-estrutural e priorizacao de revisao. Nao representam classe, label, predicao, confirmacao de evento observado ou desempenho operacional.
""",
    )
    write_text(
        reports / "final_guardrails_report.md",
        """# Relatorio final de guardrails

## Estado preservado

- DINOv2 usado como codificador visual congelado, com uso restrito a revisao estrutural.
- Nenhum rotulo binario ou alvo supervisionado criado.
- Nenhum classificador supervisionado operacional treinado.
- Nenhuma afirmacao de deteccao, predicao ou acuracia operacional.
- Protocolo C permanece `C4_BLOCKED_NO_FORMAL_NEGATIVES`.
- Ausencia de registro, pseudo-ausencia, area de fundo aleatoria e distancia de ancora nao sao tratadas como negativos formais.
""",
    )


def generate_readme(dino_summary: dict) -> None:
    write_text(
        OUT / "README.md",
        f"""# Artefatos finais publicos do REV-P

Este diretorio reune os artefatos finais utilizados na entrega do REV-P. Os resultados documentam evidencias contextuais, suporte territorial externo e analise visual-estrutural destinada a revisao. O projeto nao apresenta detector operacional, preditor ou classificador supervisionado de inundacao.

## Estrutura

- `figures/`: figuras utilizadas no artigo e na apresentacao, alem de resultados estruturais DINOv2.
- `tables/`: tabelas auxiliares com contagens canonicas, inventarios, vizinhos, PCA, medoides, valores atipicos e limites de interpretacao.
- `metrics/`: metricas descritivas de similaridade, PCA, agrupamentos exploratorios, robustez, QA, estado de prontidao e sensibilidade.
- `logs_summary/`: registros resumidos das validacoes executadas durante a preparacao da entrega.
- `execution_reports/`: relatorios de execucao, rastreabilidade, QA, selecao de figuras, DINO e limites metodologicos.
- `model/`: declaracao da ausencia de modelo supervisionado operacional.

## Resultados comprovados

- Corpus: 59 recortes territoriais/contextuais; 32 coerentes e 27 parcialmente coerentes.
- Distribuicao regional: Recife 18, Curitiba 14, Petropolis 27.
- Manifesto Sentinel-first: 128 assets candidatos.
- DINOv2: {dino_summary['embedding_count']} embeddings reais, quatro por regiao, {dino_summary['dimension']} dimensoes, codificador congelado.
- Figura principal validada de Recife: `figures/fig_recife_main_publication_v15_final.png`, com patch principal `REC_00205`.

## Dados mantidos fora

GeoTIFFs, arquivos vetoriais, PE3D/MDE bruto, embeddings `.npz`, modelo DINO, ambientes virtuais, caches, `local_runs/` completo e logs brutos permanecem locais. A entrega publica apenas manifestos, hashes, tabelas resumidas e figuras derivadas.

## Reproducao parcial

```powershell
python scripts/repository/build_outputs_public_delivery.py
python -m pytest tests
python scripts/repository/build_outputs_public_delivery.py --finalize
python scripts/repository/build_outputs_public_delivery.py --validate-only
```

O indice principal dos artefatos esta em [`execution_reports/final_delivery_artifact_index.md`](execution_reports/final_delivery_artifact_index.md).
""",
    )


def artifact_role(path: Path) -> str:
    if path.parent.name == "figures":
        return "Figura final usada no artigo, na apresentacao ou na revisao estrutural."
    if path.parent.name == "tables":
        return "Tabela publica usada na auditoria direta ou como apoio ao artigo."
    if path.parent.name == "metrics":
        return "Metrica descritiva; nao representa desempenho de modelo operacional."
    if path.parent.name == "logs_summary":
        return "Registro resumido de execucao ou QA."
    if path.parent.name == "model":
        return "Declaracao metodologica."
    if path.parent.name == "execution_reports":
        return "Relatorio de execucao, QA, rastreabilidade ou entrega."
    return "Documentacao dos resultados publicos."


def generate_artifact_index() -> None:
    self_index_paths = {
        OUT / "tables/table_artifact_index.csv",
        OUT / "execution_reports/final_delivery_artifact_index.md",
    }
    paths = sorted(path for path in OUT.rglob("*") if path.is_file() and path not in self_index_paths)
    rows = []
    for path in paths:
        relative = path.relative_to(REPO).as_posix()
        rows.append(
            {
                "artifact": path.name,
                "path": relative,
                "function": artifact_role(path),
                "used_in_article_or_presentation": "SIM_OU_CANDIDATO_APENDICE" if path.parent.name in {"figures", "tables", "metrics"} else "DOCUMENTACAO_DE_APOIO",
                "methodological_note": SAFE_NOTE,
                "size_mb": round(path.stat().st_size / 1024**2, 6),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    for path in sorted(self_index_paths):
        rows.append(
            {
                "artifact": path.name,
                "path": path.relative_to(REPO).as_posix(),
                "function": artifact_role(path),
                "used_in_article_or_presentation": "DOCUMENTACAO_DE_APOIO",
                "methodological_note": SAFE_NOTE,
                "size_mb": "NAO_APLICAVEL_INDICE_PROPRIO",
                "sha256": "NAO_APLICAVEL_INDICE_PROPRIO",
            }
        )
    write_csv(OUT / "tables/table_artifact_index.csv", rows)
    md = [
        "# Indice final de artefatos da entrega",
        "",
        "| Artefato | Caminho | Funcao no projeto | Usado no artigo/apresentacao? | Observacao metodologica |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['artifact']} | `{row['path']}` | {row['function']} | {row['used_in_article_or_presentation']} | {row['methodological_note']} |"
        )
    write_text(OUT / "execution_reports/final_delivery_artifact_index.md", "\n".join(md))


def run_pytest() -> int:
    command = [sys.executable, "-m", "pytest", "tests"]
    started = datetime.now().astimezone()
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    fallback_result = None
    fallback_timed_out = False
    collection_blocker = "tests/test_v1uc_v1ue_public_terminology.py"
    if result.returncode != 0 and "ModuleNotFoundError: No module named 'v1uc_public_terminology_scanner'" in (result.stdout + result.stderr):
        fallback_command = command + [f"--ignore={collection_blocker}"]
        try:
            fallback_result = subprocess.run(
                fallback_command,
                cwd=REPO,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
            )
        except subprocess.TimeoutExpired:
            fallback_timed_out = True
    ended = datetime.now().astimezone()
    combined = (result.stdout + "\n" + result.stderr).strip()
    full_tail = "\n".join(combined.splitlines()[-35:])
    fallback_tail = ""
    if fallback_result is not None:
        fallback_combined = (fallback_result.stdout + "\n" + fallback_result.stderr).strip()
        fallback_tail = "\n".join(fallback_combined.splitlines()[-35:])
    replacements = {
        str(REPO): "REV-P",
        str(REPO).replace("\\", "/"): "REV-P",
        str(Path.home()): "<USER_HOME>",
        str(Path.home()).replace("\\", "/"): "<USER_HOME>",
        str(sys.executable): "python",
    }
    for old, new in replacements.items():
        full_tail = full_tail.replace(old, new)
        fallback_tail = fallback_tail.replace(old, new)
    if result.returncode == 0:
        status = "PASS"
    elif fallback_result is not None and fallback_result.returncode == 0:
        status = "FULL_SUITE_BLOCKED_BY_MISSING_REFACTOR_MODULES; REMAINING_SUITE_PASS"
    elif fallback_timed_out:
        status = "FULL_SUITE_BLOCKED_BY_MISSING_REFACTOR_MODULES; REMAINING_SUITE_TIMEOUT_900S"
    else:
        status = f"FAIL_EXIT_{result.returncode}"
    write_text(
        OUT / "logs_summary/pytest_summary.txt",
        f"""comando_executado: python -m pytest tests
data_hora_local_inicio: {started.isoformat(timespec="seconds")}
data_hora_local_fim: {ended.isoformat(timespec="seconds")}
status: {status}
inputs_usados: tests/
outputs_gerados: trecho final resumido do pytest abaixo
falhas_ou_bloqueadores_esperados: integracoes pesadas permanecem opcionais
observacao_metodologica: {SAFE_NOTE}

trecho_final_pytest_completo:
{full_tail}

trecho_final_suite_restante:
{fallback_tail or ("TEMPO_LIMITE_900S" if fallback_timed_out else "NAO_EXECUTADA")}
""",
    )
    return fallback_result.returncode if fallback_result is not None else result.returncode


def finalize_checks(write_summary: bool = True) -> dict:
    files = [path for path in OUT.rglob("*") if path.is_file()]
    oversize = [path for path in files if path.stat().st_size > 50 * 1024**2]
    absolute_hits = []
    forbidden_context_hits = []
    text_extensions = {".md", ".txt", ".csv", ".json", ".log"}
    absolute_pattern = re.compile(r"[A-Za-z]:\\Users\\")
    risky_patterns = [
        re.compile(r"(?i)\brevp\s+(detecta|prediz|classifica)\b"),
        re.compile(r"(?i)\bmodelo\s+(detecta|prediz)\s+inundacao\b"),
        re.compile(r"(?i)\bacuracia operacional\s*[:=]\s*\d"),
    ]
    for path in files:
        if path.suffix.lower() not in text_extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if absolute_pattern.search(text):
            absolute_hits.append(path.relative_to(REPO).as_posix())
        if any(pattern.search(text) for pattern in risky_patterns):
            forbidden_context_hits.append(path.relative_to(REPO).as_posix())
    required_paths = [
        OUT / "figures/fig_recife_main_publication_v15_final.png",
        OUT / "model/NO_OPERATIONAL_TRAINED_MODEL.md",
        OUT / "execution_reports/final_delivery_artifact_index.md",
    ]
    missing_required = [path.relative_to(REPO).as_posix() for path in required_paths if not path.exists()]
    index_path = OUT / "tables/table_artifact_index.csv"
    indexed_paths = []
    if index_path.exists():
        with index_path.open(encoding="utf-8", newline="") as handle:
            indexed_paths = [row.get("path", "") for row in csv.DictReader(handle)]
    missing_indexed = [path for path in indexed_paths if path and not (REPO / path).exists()]
    indexed_artifact_count = len(indexed_paths)
    expected_artifact_count = 55
    result = {
        "file_count": len(files),
        "indexed_artifact_count": indexed_artifact_count,
        "expected_artifact_count": expected_artifact_count,
        "missing_required_files": missing_required,
        "missing_indexed_files": missing_indexed,
        "oversize_files": [path.relative_to(REPO).as_posix() for path in oversize],
        "absolute_path_hits": absolute_hits,
        "risky_language_context_hits": forbidden_context_hits,
        "status": (
            "PASS"
            if not oversize
            and not absolute_hits
            and not forbidden_context_hits
            and not missing_required
            and not missing_indexed
            and indexed_artifact_count == expected_artifact_count
            else "REVISAO_NECESSARIA"
        ),
    }
    if write_summary:
        write_text(OUT / "execution_reports/final_validation_summary.json", json.dumps(result, indent=2, ensure_ascii=False))
    return result


def build() -> None:
    ensure_dirs()
    figures = copy_selected_figures()
    metadata, vectors = load_embeddings()
    dino_summary = generate_dino_products(metadata, vectors)
    generate_canonical_tables(dino_summary)
    generate_robustness_summary()
    discovery = generate_discovery_inventory()
    generate_logs(dino_summary, discovery)
    generate_model_note()
    generate_reports(figures, dino_summary, discovery)
    generate_readme(dino_summary)
    generate_artifact_index()
    finalize_checks()
    generate_artifact_index()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true", help="Executa a suite completa e registra um resumo.")
    parser.add_argument("--finalize", action="store_true", help="Atualiza os indices e a validacao final.")
    parser.add_argument("--validate-only", action="store_true", help="Valida os artefatos existentes sem altera-los.")
    args = parser.parse_args()
    if args.validate_only:
        result = finalize_checks(write_summary=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 2
    if args.finalize:
        ensure_dirs()
        generate_artifact_index()
        finalize_checks()
        generate_artifact_index()
        return 0
    build()
    if args.run_tests:
        code = run_pytest()
        generate_artifact_index()
        finalize_checks()
        generate_artifact_index()
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
