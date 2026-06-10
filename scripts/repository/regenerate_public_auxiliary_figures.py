"""Regenerate REV-P public auxiliary figures from public summary artifacts.

The figures remain review-only and do not represent operational validation,
prediction, classification, or patch-level ground truth.
"""
from __future__ import annotations

import csv
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_public"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
METRICS = OUT / "metrics"
AUDIT_ORIGINALS = OUT / "appendix_visual_audit" / "original_auxiliary_figures"
REPORT = OUT / "execution_reports" / "figures_regeneration_report.md"
MANIFEST = TABLES / "figure_regeneration_manifest.csv"

REGION_ORDER = ["Recife", "Petropolis", "Curitiba"]
REGION_LABEL = {"Recife": "Recife", "Petropolis": "Petr\u00f3polis", "Curitiba": "Curitiba"}
REGION_COLOR = {"Recife": "#177E89", "Petropolis": "#C66A3D", "Curitiba": "#5968A8"}
NAVY = "#203040"
INK = "#263238"
MUTED = "#60717A"
GRID = "#DCE4E8"
PALE = "#F5F8FA"
GREEN = "#3C8D6F"
AMBER = "#D29B42"
RED = "#B85C5C"

MAIN_FIGURES = [
    "fig_recife_main_publication_v15_final.png",
    "fig_curitiba_main_publication_v17.png",
    "fig_petropolis_main_publication_v17.png",
    "fig_recife_pe3d_mde_publication.png",
    "fig_recife_sentinel_technical_publication.png",
]

AUXILIARY_FIGURES = [
    "dino_knn_neighbor_network_publication.png",
    "dino_medoids_outliers_publication.png",
    "dino_pca_projection_publication.png",
    "dino_region_neighbor_matrix_publication.png",
    "dino_similarity_heatmap_publication.png",
    "fig_corpus_counts_by_region_status.png",
    "fig_decision_trace_summary.png",
    "fig_dino_input_corpus_publication.png",
    "fig_evidence_layer_availability.png",
    "fig_local_context_coverage.png",
    "fig_methodological_contribution_matrix.png",
    "fig_regional_roles_summary.png",
]

SOURCE_MAP = {
    "dino_knn_neighbor_network_publication.png": "outputs_public/tables/table_dino_nearest_neighbors.csv",
    "dino_medoids_outliers_publication.png": "outputs_public/tables/table_dino_medoids.csv | outputs_public/tables/table_dino_outliers.csv",
    "dino_pca_projection_publication.png": "outputs_public/tables/table_dino_pca_coordinates.csv | outputs_public/tables/table_dino_medoids.csv | outputs_public/tables/table_dino_outliers.csv | outputs_public/metrics/dino_pca_summary.csv",
    "dino_region_neighbor_matrix_publication.png": "outputs_public/tables/table_dino_region_neighbor_matrix.csv",
    "dino_similarity_heatmap_publication.png": "outputs_public/tables/table_dino_similarity_matrix.csv",
    "fig_corpus_counts_by_region_status.png": "outputs_public/tables/table_patch_distribution_by_region.csv | outputs_public/tables/table_corpus_summary.csv",
    "fig_decision_trace_summary.png": "outputs_public/tables/table_protocol_c_summary.csv | outputs_public/metrics/readiness_summary.csv",
    "fig_dino_input_corpus_publication.png": "outputs_public/tables/table_dino_embedding_inventory.csv | outputs_public/tables/table_dino_quantitative_summary_by_region.csv",
    "fig_evidence_layer_availability.png": "outputs_public/tables/table_external_evidence_summary.csv",
    "fig_local_context_coverage.png": "outputs_public/tables/table_external_evidence_summary.csv | outputs_public/tables/table_patch_distribution_by_region.csv | outputs_public/metrics/readiness_summary.csv",
    "fig_methodological_contribution_matrix.png": "outputs_public/README.md | outputs_public/tables/table_claims_guardrails_summary.csv | outputs_public/metrics/qa_metrics_summary.csv | outputs_public/metrics/readiness_summary.csv",
    "fig_regional_roles_summary.png": "outputs_public/tables/table_patch_distribution_by_region.csv | outputs_public/tables/table_external_evidence_summary.csv",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def read_table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_metric(name: str) -> pd.DataFrame:
    path = METRICS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def region_from_patch(patch_id: str) -> str:
    prefix = str(patch_id).split("_", 1)[0].upper()
    return {"REC": "Recife", "PET": "Petropolis", "CUR": "Curitiba"}[prefix]


def short_patch(patch_id: str) -> str:
    return str(patch_id).replace("_", "\n", 1)


def add_title(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.suptitle(title, x=0.06, y=0.975, ha="left", color=NAVY, fontsize=18, fontweight="bold")
    if subtitle:
        fig.text(0.06, 0.925, subtitle, ha="left", color=MUTED, fontsize=10)


def add_note(fig: plt.Figure, note: str) -> None:
    fig.text(0.06, 0.025, note, ha="left", va="bottom", color=MUTED, fontsize=8.5)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def backup_originals() -> None:
    AUDIT_ORIGINALS.mkdir(parents=True, exist_ok=True)
    for name in AUXILIARY_FIGURES:
        source = FIGURES / name
        destination = AUDIT_ORIGINALS / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)


def similarity_heatmap() -> None:
    df = read_table("table_dino_similarity_matrix.csv").set_index("patch_id")
    order = sorted(df.index, key=lambda patch: (REGION_ORDER.index(region_from_patch(patch)), patch))
    matrix = df.loc[order, order].astype(float)
    fig, ax = plt.subplots(figsize=(11.2, 9.3))
    add_title(
        fig,
        "Similaridade visual-estrutural entre embeddings DINOv2",
        "Encoder congelado; an\u00e1lise explorat\u00f3ria para revis\u00e3o humana",
    )
    im = ax.imshow(matrix, cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0.3, vmax=1.0)
    labels = [patch.replace("_", " ") for patch in order]
    ax.set_xticks(range(len(order)), labels=labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), labels=labels, fontsize=8)
    ax.tick_params(length=0)
    for boundary in [4, 8]:
        ax.axhline(boundary - 0.5, color="white", lw=2.2)
        ax.axvline(boundary - 0.5, color="white", lw=2.2)
    for i in range(len(order)):
        for j in range(len(order)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.3, color="white" if value < 0.72 else NAVY)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Similaridade cosseno", color=INK)
    add_note(fig, "A matriz descreve proximidade visual-estrutural; n\u00e3o representa classe, predi\u00e7\u00e3o ou ground truth.")
    fig.subplots_adjust(top=0.86, bottom=0.16)
    save(fig, "dino_similarity_heatmap_publication.png")


def region_neighbor_matrix() -> None:
    df = read_table("table_dino_region_neighbor_matrix.csv")
    pivot = df.pivot(index="query_region", columns="neighbor_region", values="edge_share").reindex(index=REGION_ORDER, columns=REGION_ORDER)
    fig, ax = plt.subplots(figsize=(8.8, 7.3))
    add_title(fig, "Vizinhan\u00e7a visual-estrutural agregada por regi\u00e3o")
    im = ax.imshow(pivot.astype(float), cmap=mcolors.LinearSegmentedColormap.from_list("revp", ["#EEF4F5", "#8BC0C3", "#176B70"]), vmin=0, vmax=1)
    labels = [REGION_LABEL[r] for r in REGION_ORDER]
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Regi\u00e3o dos vizinhos", labelpad=12)
    ax.set_ylabel("Regi\u00e3o de consulta", labelpad=12)
    for i in range(3):
        for j in range(3):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=14, fontweight="bold", color="white" if value > 0.55 else NAVY)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Participa\u00e7\u00e3o das arestas")
    add_note(fig, "Matriz derivada dos embeddings DINOv2; n\u00e3o representa valida\u00e7\u00e3o operacional.")
    fig.subplots_adjust(top=0.86, bottom=0.13)
    save(fig, "dino_region_neighbor_matrix_publication.png")


def pca_projection() -> None:
    df = read_table("table_dino_pca_coordinates.csv")
    medoids = set(read_table("table_dino_medoids.csv")["patch_id"])
    outliers = set(read_table("table_dino_outliers.csv")["patch_id"])
    pca_summary = read_metric("dino_pca_summary.csv").set_index("component")
    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    add_title(fig, "Proje\u00e7\u00e3o PCA dos embeddings DINOv2", "Visualiza\u00e7\u00e3o explorat\u00f3ria do espa\u00e7o latente")
    for region in REGION_ORDER:
        subset = df[df["region"] == region]
        ax.scatter(subset["pca_1"], subset["pca_2"], s=110, color=REGION_COLOR[region], edgecolor="white", linewidth=1.5, label=REGION_LABEL[region], zorder=3)
    for _, row in df.iterrows():
        patch = row["patch_id"]
        if patch in medoids or patch in outliers:
            marker = "*" if patch in medoids else "X"
            ax.scatter(row["pca_1"], row["pca_2"], s=260 if marker == "*" else 170, facecolor="none", edgecolor=NAVY, marker=marker, linewidth=1.7, zorder=4)
            ax.annotate(patch, (row["pca_1"], row["pca_2"]), xytext=(6, 7), textcoords="offset points", fontsize=8, color=INK)
    pca1 = float(pca_summary.loc["PCA1", "explained_variance_ratio"])
    pca2 = float(pca_summary.loc["PCA2", "explained_variance_ratio"])
    ax.set_xlabel(f"PCA 1 ({pca1:.1%} da vari\u00e2ncia)")
    ax.set_ylabel(f"PCA 2 ({pca2:.1%} da vari\u00e2ncia)")
    clean_axes(ax, "both")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], marker="*", color=NAVY, markerfacecolor="none", markersize=12, linestyle="None", label="Medoid"),
            Line2D([], [], marker="X", color=NAVY, markerfacecolor="none", markersize=9, linestyle="None", label="Outlier"),
        ]
    )
    ax.legend(handles=handles, frameon=False, loc="best", ncol=2)
    add_note(fig, "A distribui\u00e7\u00e3o no espa\u00e7o latente n\u00e3o indica separa\u00e7\u00e3o de classes.")
    fig.subplots_adjust(top=0.86, bottom=0.12)
    save(fig, "dino_pca_projection_publication.png")


def knn_network() -> None:
    df = read_table("table_dino_nearest_neighbors.csv")
    df = df[df["neighbor_rank"] <= 2].copy()
    nodes = sorted(set(df["query_patch_id"]) | set(df["neighbor_patch_id"]))
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node, region=region_from_patch(node))
    for _, row in df.iterrows():
        a, b = row["query_patch_id"], row["neighbor_patch_id"]
        weight = float(row["cosine_similarity"])
        if graph.has_edge(a, b):
            graph[a][b]["weight"] = max(graph[a][b]["weight"], weight)
        else:
            graph.add_edge(a, b, weight=weight)
    positions = {}
    x_by_region = {"Recife": -1.5, "Petropolis": 0.0, "Curitiba": 1.5}
    for region in REGION_ORDER:
        region_nodes = sorted(node for node in nodes if graph.nodes[node]["region"] == region)
        for i, node in enumerate(region_nodes):
            angle = 2 * math.pi * i / len(region_nodes) + math.pi / 4
            positions[node] = (x_by_region[region] + 0.48 * math.cos(angle), 0.75 * math.sin(angle))
    fig, ax = plt.subplots(figsize=(12.2, 7.8))
    add_title(fig, "Rede de vizinhan\u00e7a entre patches no espa\u00e7o DINOv2")
    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, positions, ax=ax, width=[0.7 + 2.2 * max(0, w - 0.7) / 0.25 for w in weights], edge_color="#A8B6BC", alpha=0.65)
    for region in REGION_ORDER:
        region_nodes = [node for node in nodes if graph.nodes[node]["region"] == region]
        nx.draw_networkx_nodes(graph, positions, nodelist=region_nodes, node_color=REGION_COLOR[region], node_size=720, edgecolors="white", linewidths=1.6, ax=ax)
    nx.draw_networkx_labels(graph, positions, labels={node: short_patch(node) for node in nodes}, font_size=7.2, font_color="white", ax=ax)
    for region in REGION_ORDER:
        ax.text(x_by_region[region], 1.13, REGION_LABEL[region], ha="center", color=REGION_COLOR[region], fontsize=12, fontweight="bold")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.1, 1.3)
    ax.axis("off")
    ax.legend(handles=[Patch(color=REGION_COLOR[r], label=REGION_LABEL[r]) for r in REGION_ORDER], frameon=False, loc="lower center", ncol=3)
    add_note(fig, "Arestas mostram os dois vizinhos mais pr\u00f3ximos por patch; indicam proximidade visual-estrutural, n\u00e3o rela\u00e7\u00e3o causal ou valida\u00e7\u00e3o de evento.")
    fig.subplots_adjust(top=0.86, bottom=0.12)
    save(fig, "dino_knn_neighbor_network_publication.png")


def medoids_outliers() -> None:
    medoids = read_table("table_dino_medoids.csv")
    outliers = read_table("table_dino_outliers.csv")
    scopes = ["Curitiba", "Petropolis", "Recife", "Corpus"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.4))
    add_title(fig, "Medoids e outliers no espa\u00e7o visual-estrutural", "Patches de refer\u00eancia e casos priorit\u00e1rios para revis\u00e3o humana")
    for ax, frame, heading, color in [(axes[0], medoids, "Medoids", GREEN), (axes[1], outliers, "Outliers", AMBER)]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.03, 0.96, heading, va="top", color=color, fontsize=15, fontweight="bold")
        for i, scope in enumerate(scopes):
            row = frame[frame["scope"] == scope].iloc[0]
            y = 0.78 - i * 0.19
            box = FancyBboxPatch((0.03, y - 0.07), 0.92, 0.135, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=PALE, edgecolor=GRID)
            ax.add_patch(box)
            ax.text(0.06, y + 0.025, REGION_LABEL.get(scope, scope), color=MUTED, fontsize=9, fontweight="bold")
            ax.text(0.06, y - 0.022, row["patch_id"], color=NAVY, fontsize=12, fontweight="bold")
            metric = float(row["mean_similarity_within_scope"])
            ax.text(0.91, y - 0.005, f"{metric:.3f}", ha="right", color=color, fontsize=13, fontweight="bold")
            ax.text(0.91, y - 0.04, "similaridade m\u00e9dia", ha="right", color=MUTED, fontsize=7.5)
    add_note(fig, "Medoids resumem casos centrais e outliers sinalizam casos distintos; ambos servem \u00e0 prioriza\u00e7\u00e3o de revis\u00e3o humana.")
    fig.subplots_adjust(top=0.85, bottom=0.11, wspace=0.12)
    save(fig, "dino_medoids_outliers_publication.png")


def corpus_counts() -> None:
    df = read_table("table_patch_distribution_by_region.csv").set_index("region").reindex(REGION_ORDER)
    coherent = df["coherent_count"].astype(int)
    partial = df["partially_coherent_count"].astype(int)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    add_title(fig, "Distribui\u00e7\u00e3o dos patches por regi\u00e3o e status")
    bars1 = ax.bar(x, coherent, width=0.58, color=GREEN, label="Coer\u00eancia externa")
    bars2 = ax.bar(x, partial, width=0.58, bottom=coherent, color=AMBER, label="Coer\u00eancia parcial")
    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_height())}", ha="center", va="center", color="white", fontweight="bold")
    totals = coherent + partial
    for i, total in enumerate(totals):
        ax.text(i, total + 0.8, f"{total} patches", ha="center", color=NAVY, fontweight="bold")
    ax.set_xticks(x, [REGION_LABEL[r] for r in REGION_ORDER])
    ax.set_ylabel("Quantidade de patches")
    ax.set_ylim(0, max(totals) + 5)
    clean_axes(ax, "y")
    ax.legend(frameon=False, loc="upper right")
    add_note(fig, "Os status descrevem coer\u00eancia contextual externa e n\u00e3o constituem classes operacionais.")
    fig.subplots_adjust(top=0.86, bottom=0.12)
    save(fig, "fig_corpus_counts_by_region_status.png")


def decision_trace() -> None:
    protocol = read_table("table_protocol_c_summary.csv")
    readiness = read_metric("readiness_summary.csv")
    statuses = list(protocol["status"].astype(str)) + list(readiness["status"].astype(str))
    categories = {"Pronto para revis\u00e3o": 0, "Uso restrito": 0, "Bloqueado": 0}
    for status in statuses:
        upper = status.upper()
        if "BLOCKED" in upper:
            categories["Bloqueado"] += 1
        elif "RESTRITO" in upper:
            categories["Uso restrito"] += 1
        else:
            categories["Pronto para revis\u00e3o"] += 1
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    add_title(fig, "Trilha de decis\u00e3o metodol\u00f3gica", "Resumo dos estados de revis\u00e3o e bloqueio operacional")
    labels = list(categories)
    values = list(categories.values())
    colors = [GREEN, AMBER, RED]
    bars = ax.barh(labels, values, color=colors, height=0.55)
    for bar, value in zip(bars, values):
        ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2, str(value), va="center", color=NAVY, fontsize=13, fontweight="bold")
    ax.set_xlabel("Registros de decis\u00e3o documentados")
    ax.set_xlim(0, max(values) + 1.5)
    clean_axes(ax, "x")
    ax.invert_yaxis()
    ax.text(0.99, 0.04, f"{len(statuses)} estados consolidados", transform=ax.transAxes, ha="right", color=MUTED, fontsize=9)
    add_note(fig, "Estados indicam uso permitido da evid\u00eancia; n\u00e3o s\u00e3o classes de inunda\u00e7\u00e3o.")
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.25)
    save(fig, "fig_decision_trace_summary.png")


def dino_input_corpus() -> None:
    inventory = read_table("table_dino_embedding_inventory.csv")
    summary = read_table("table_dino_quantitative_summary_by_region.csv").set_index("region")
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 6.9))
    add_title(fig, "Corpus de entrada para o DINOv2", "12 patches com embeddings reais, quatro por regi\u00e3o")
    for ax, region in zip(axes, REGION_ORDER):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.88, boxstyle="round,pad=0.015,rounding_size=0.03", facecolor=PALE, edgecolor=GRID))
        ax.add_patch(FancyBboxPatch((0.02, 0.79), 0.96, 0.14, boxstyle="round,pad=0.015,rounding_size=0.03", facecolor=REGION_COLOR[region], edgecolor=REGION_COLOR[region]))
        ax.text(0.07, 0.86, REGION_LABEL[region], color="white", fontsize=14, fontweight="bold", va="center")
        patches = inventory[inventory["region"] == region]["patch_id"].tolist()
        for i, patch in enumerate(patches):
            y = 0.69 - i * 0.13
            ax.text(0.09, y, patch, color=NAVY, fontsize=10.5, fontweight="bold")
            ax.text(0.91, y, "768D", color=MUTED, fontsize=9, ha="right")
            ax.plot([0.09, 0.91], [y - 0.045, y - 0.045], color=GRID, lw=0.8)
        ax.text(0.09, 0.12, f"Similaridade intrarregional m\u00e9dia: {float(summary.loc[region, 'mean_intra_region_similarity']):.3f}", color=MUTED, fontsize=8.5)
    add_note(fig, "Corpus curado e balanceado para an\u00e1lise visual-estrutural com encoder congelado; sem alvo supervisionado.")
    fig.subplots_adjust(top=0.83, bottom=0.11, wspace=0.08)
    save(fig, "fig_dino_input_corpus_publication.png")


def evidence_statuses() -> tuple[pd.DataFrame, pd.DataFrame]:
    external = read_table("table_external_evidence_summary.csv").set_index("region").reindex(REGION_ORDER)
    rows = ["Relevo", "Suscetibilidade externa", "Hidrografia"]
    matrix = pd.DataFrame(0, index=rows, columns=REGION_ORDER, dtype=int)
    for region, row in external.iterrows():
        support = str(row["support"]).lower()
        matrix.loc["Relevo", region] = int("relevo" in support or "pe3d" in support)
        matrix.loc["Suscetibilidade externa", region] = int("suscetibilidade" in support)
        matrix.loc["Hidrografia", region] = int("hidrografia" in support)
    return external, matrix


def evidence_availability() -> None:
    _, matrix = evidence_statuses()
    fig, ax = plt.subplots(figsize=(9.8, 6.7))
    add_title(fig, "Disponibilidade de evid\u00eancias externas por regi\u00e3o")
    cmap = mcolors.ListedColormap(["#EDF1F3", "#4F9B83"])
    ax.imshow(matrix.values, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(3), [REGION_LABEL[r] for r in REGION_ORDER])
    ax.set_yticks(range(len(matrix)), matrix.index)
    for i in range(len(matrix)):
        for j in range(3):
            documented = int(matrix.iloc[i, j]) == 1
            ax.text(j, i, "Documentado" if documented else "N\u00e3o indicado\nno resumo", ha="center", va="center", fontsize=9, color="white" if documented else MUTED, fontweight="bold" if documented else "normal")
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(color="#4F9B83", label="Suporte documentado"), Patch(color="#EDF1F3", label="N\u00e3o indicado no resumo p\u00fablico")], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    add_note(fig, "Camadas externas apoiam interpreta\u00e7\u00e3o contextual; n\u00e3o substituem ground truth.")
    fig.subplots_adjust(top=0.83, bottom=0.22, left=0.24)
    save(fig, "fig_evidence_layer_availability.png")


def local_context_coverage() -> None:
    _, matrix = evidence_statuses()
    patches = read_table("table_patch_distribution_by_region.csv").set_index("region").reindex(REGION_ORDER)["patch_count"].astype(int)
    documented = matrix.sum(axis=0).reindex(REGION_ORDER)
    fig, ax = plt.subplots(figsize=(10.6, 7.2))
    add_title(fig, "Cobertura contextual dos recortes regionais", "Suportes explicitamente documentados no resumo p\u00fablico")
    x = np.arange(3)
    bars = ax.bar(x, documented, width=0.55, color=[REGION_COLOR[r] for r in REGION_ORDER])
    for i, (bar, count) in enumerate(zip(bars, patches)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, f"{int(bar.get_height())} tipos documentados", ha="center", color=NAVY, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.18, f"{count} patches", ha="center", color="white", fontsize=10, fontweight="bold")
    ax.set_xticks(x, [REGION_LABEL[r] for r in REGION_ORDER])
    ax.set_ylabel("Tipos de suporte contextual no resumo")
    ax.set_ylim(0, max(documented) + 1.1)
    ax.set_yticks(range(0, int(max(documented)) + 2))
    clean_axes(ax, "y")
    add_note(fig, "A contagem descreve o resumo p\u00fablico dispon\u00edvel e n\u00e3o mede completude, qualidade ou valida\u00e7\u00e3o operacional.")
    fig.subplots_adjust(top=0.84, bottom=0.13)
    save(fig, "fig_local_context_coverage.png")


def methodological_matrix() -> None:
    rows = [
        ("Corpus", "Organiza recortes", "59 patches contextuais", "Sem label operacional"),
        ("Sentinel", "Estrutura invent\u00e1rio", "128 ativos candidatos", "Sem confirma\u00e7\u00e3o de evento"),
        ("Evid\u00eancias externas", "Apoia interpreta\u00e7\u00e3o", "Suporte territorial regional", "N\u00e3o substitui ground truth"),
        ("QA e rastreabilidade", "Verifica consist\u00eancia", "Contagens e artefatos auditados", "N\u00e3o mede acur\u00e1cia operacional"),
        ("Protocolo C", "Controla claims", "Transi\u00e7\u00e3o operacional bloqueada", "Sem negativos formais"),
        ("DINOv2", "Explora estrutura visual", "12 embeddings reais, 768D", "Encoder congelado; revis\u00e3o"),
    ]
    columns = ["Etapa", "Fun\u00e7\u00e3o", "Resultado", "Limite de interpreta\u00e7\u00e3o"]
    fig, ax = plt.subplots(figsize=(14.2, 7.8))
    add_title(fig, "Contribui\u00e7\u00f5es metodol\u00f3gicas do REV-P")
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left", loc="center", colWidths=[0.16, 0.22, 0.28, 0.30])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F3F6F7" if row % 2 else "#E8EFF1")
            if col == 0:
                cell.set_text_props(color=NAVY, fontweight="bold")
    add_note(fig, "S\u00edntese baseada nos artefatos p\u00fablicos, nos guardrails de claims e nas m\u00e9tricas de QA/prontid\u00e3o.")
    fig.subplots_adjust(top=0.84, bottom=0.1)
    save(fig, "fig_methodological_contribution_matrix.png")


def regional_roles() -> None:
    distribution = read_table("table_patch_distribution_by_region.csv").set_index("region")
    evidence = read_table("table_external_evidence_summary.csv").set_index("region")
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 7.2))
    add_title(fig, "Papel das regi\u00f5es no corpus REV-P")
    for ax, region in zip(axes, REGION_ORDER):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.9, boxstyle="round,pad=0.015,rounding_size=0.03", facecolor=PALE, edgecolor=GRID))
        ax.add_patch(FancyBboxPatch((0.02, 0.79), 0.96, 0.15, boxstyle="round,pad=0.015,rounding_size=0.03", facecolor=REGION_COLOR[region], edgecolor=REGION_COLOR[region]))
        ax.text(0.07, 0.865, REGION_LABEL[region], color="white", fontsize=15, fontweight="bold", va="center")
        ax.text(0.07, 0.72, f"{int(distribution.loc[region, 'patch_count'])}", color=NAVY, fontsize=27, fontweight="bold")
        ax.text(0.25, 0.735, "patches contextuais", color=MUTED, fontsize=9, va="center")
        ax.text(0.07, 0.61, "Papel regional", color=REGION_COLOR[region], fontsize=9, fontweight="bold")
        ax.text(0.07, 0.55, textwrap.fill(str(evidence.loc[region, "role"]), 34), color=INK, fontsize=9.5, va="top")
        ax.text(0.07, 0.38, "Suporte territorial", color=REGION_COLOR[region], fontsize=9, fontweight="bold")
        ax.text(0.07, 0.32, textwrap.fill(str(evidence.loc[region, "support"]), 34), color=INK, fontsize=9.5, va="top")
        caution = {
            "Recife": "Suporte contextual; sem ground truth operacional.",
            "Petropolis": "Suporte parcial; sem reclassifica\u00e7\u00e3o dos recortes.",
            "Curitiba": "Contraste metodol\u00f3gico; n\u00e3o cria negativos formais.",
        }[region]
        ax.text(0.07, 0.16, "Cautela metodol\u00f3gica", color=REGION_COLOR[region], fontsize=9, fontweight="bold")
        ax.text(0.07, 0.105, textwrap.fill(caution, 36), color=MUTED, fontsize=8.5, va="top")
    add_note(fig, "Os pap\u00e9is regionais s\u00e3o complementares e n\u00e3o estabelecem hierarquia de evid\u00eancia.")
    fig.subplots_adjust(top=0.83, bottom=0.1, wspace=0.07)
    save(fig, "fig_regional_roles_summary.png")


def write_manifest_and_report() -> None:
    rows: list[dict[str, str]] = []
    for name in MAIN_FIGURES:
        rows.append(
            {
                "figure_path": f"outputs_public/figures/{name}",
                "status": "preserved_main_figure",
                "source_data": "Figura principal preservada sem alteracao.",
                "regenerated": "false",
                "visual_problem_detected": "Nao avaliado para regeneracao; fora do escopo.",
                "change_summary": "Nenhuma alteracao.",
                "scientific_role": "Figura principal de patch, regiao ou suporte tecnico.",
                "safe_interpretation": "Curadoria visual principal no escopo documentado pelo REV-P.",
                "forbidden_interpretation": "Nao extrapolar para validacao operacional ou ground truth.",
                "notes": "Preservada por regra explicita da regeneracao.",
            }
        )
    problems = {
        "dino_knn_neighbor_network_publication.png": "Rede com leitura de debug e excesso de arestas/rotulos.",
        "dino_medoids_outliers_publication.png": "Resumo cru, sem separacao editorial entre medoids e outliers.",
        "dino_pca_projection_publication.png": "Projecao com hierarquia visual e anotacoes pouco consistentes.",
        "dino_region_neighbor_matrix_publication.png": "Matriz com acabamento e notas metodologicas insuficientes.",
        "dino_similarity_heatmap_publication.png": "Heatmap com leitura editorial limitada.",
        "fig_corpus_counts_by_region_status.png": "Contagens regionais pouco comunicativas.",
        "fig_decision_trace_summary.png": "Rotulos tecnicos extensos e baixa comunicacao dos estados.",
        "fig_dino_input_corpus_publication.png": "Entrada DINOv2 sem sintese visual clara do balanceamento.",
        "fig_evidence_layer_availability.png": "Disponibilidade regional sem matriz comunicavel.",
        "fig_local_context_coverage.png": "Cobertura contextual sem sintese regional direta.",
        "fig_methodological_contribution_matrix.png": "Matriz metodologica com aparencia pouco editorial.",
        "fig_regional_roles_summary.png": "Papeis regionais sem painel comparativo conciso.",
    }
    roles = {
        "dino_knn_neighbor_network_publication.png": "Vizinhança visual-estrutural para revisão.",
        "dino_medoids_outliers_publication.png": "Priorização de revisão por centralidade e distinção.",
        "dino_pca_projection_publication.png": "Visualização exploratória do espaço latente.",
        "dino_region_neighbor_matrix_publication.png": "Síntese regional das vizinhanças DINOv2.",
        "dino_similarity_heatmap_publication.png": "Síntese da similaridade cosseno entre embeddings.",
        "fig_corpus_counts_by_region_status.png": "Estrutura regional do corpus contextual.",
        "fig_decision_trace_summary.png": "Síntese dos estados metodológicos.",
        "fig_dino_input_corpus_publication.png": "Documentação do corpus curado de entrada DINOv2.",
        "fig_evidence_layer_availability.png": "Documentação do suporte externo resumido.",
        "fig_local_context_coverage.png": "Síntese dos suportes contextuais documentados.",
        "fig_methodological_contribution_matrix.png": "Síntese das contribuições e limites do REV-P.",
        "fig_regional_roles_summary.png": "Comparação dos papéis regionais no corpus.",
    }
    for name in AUXILIARY_FIGURES:
        rows.append(
            {
                "figure_path": f"outputs_public/figures/{name}",
                "status": "regenerated_auxiliary",
                "source_data": SOURCE_MAP[name],
                "regenerated": "true",
                "visual_problem_detected": problems[name],
                "change_summary": "Regenerada com estilo editorial consistente, rotulos em portugues e nota metodologica.",
                "scientific_role": roles[name],
                "safe_interpretation": "Resultado contextual ou visual-estrutural destinado a revisao humana.",
                "forbidden_interpretation": "Nao interpretar como classe, predicao, deteccao, validacao de evento ou ground truth.",
                "notes": f"Versao anterior preservada em outputs_public/appendix_visual_audit/original_auxiliary_figures/{name}.",
            }
        )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_lines = "\n".join(f"- `{name}`: `{SOURCE_MAP[name].replace(' | ', '`, `')}`." for name in AUXILIARY_FIGURES)
    report = f"""# Relatorio de regeneracao das figuras auxiliares

## Objetivo

As figuras auxiliares de `outputs_public/figures/` foram regeneradas para melhorar leitura, consistencia visual e comunicacao metodologica em GitHub, PDF e apresentacoes. A regeneracao utilizou somente tabelas, metricas e relatorios publicos existentes.

## Figuras preservadas

As cinco figuras principais definidas como fora do escopo foram preservadas sem sobrescrita:

{chr(10).join(f"- `outputs_public/figures/{name}`" for name in MAIN_FIGURES)}

## Figuras regeneradas

Foram regeneradas {len(AUXILIARY_FIGURES)} figuras auxiliares, mantendo os caminhos publicos. As versoes anteriores foram copiadas para `outputs_public/appendix_visual_audit/original_auxiliary_figures/`.

## Dados usados

{source_lines}

## Problemas visuais corrigidos

- Rotulos tecnicos extensos foram substituidos por portugues comunicavel.
- Titulos, subtitulos, notas, margens, cores e hierarquia tipografica foram padronizados.
- Matrizes e graficos DINOv2 passaram a explicitar o papel exploratorio e review-only.
- Resumos de corpus, decisao, evidencias, contribuicoes e papeis regionais foram convertidos em composicoes editoriais mais diretas.

## Guardrails mantidos

- Nenhuma figura apresenta DINOv2 como classificador, detector ou preditor.
- Nenhuma figura sugere ground truth operacional, label binario, classe positiva/negativa ou acuracia operacional.
- A analise DINOv2 permanece baseada em encoder congelado, embeddings 768D e revisao humana.
- A validacao operacional e a transicao C4 permanecem bloqueadas conforme os artefatos publicos.
- Nenhum dado bruto pesado, GeoTIFF, shapefile, `.npz`, modelo, cache ou copia de `local_runs/` foi incluido.
- Os arquivos nao atribuem autoria a ferramentas automaticas.

## Rastreabilidade

O manifesto `outputs_public/tables/figure_regeneration_manifest.csv` registra o status, os dados de origem, o papel cientifico e os limites de interpretacao de cada figura principal preservada e auxiliar regenerada.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    setup_style()
    backup_originals()
    similarity_heatmap()
    region_neighbor_matrix()
    pca_projection()
    knn_network()
    medoids_outliers()
    corpus_counts()
    decision_trace()
    dino_input_corpus()
    evidence_availability()
    local_context_coverage()
    methodological_matrix()
    regional_roles()
    write_manifest_and_report()
    print(f"Regenerated {len(AUXILIARY_FIGURES)} auxiliary figures.")


if __name__ == "__main__":
    main()
