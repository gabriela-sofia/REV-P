"""REV-P v1gy: TCC visual evidence export package.

Reads local outputs from v1gu/v1gv/v1gw/v1gx and produces:
- publication-ready figures (PNG, matplotlib)
- clean CSV tables for LaTeX conversion
- artifact manifest with status, captions, and limitations
- summary JSON with counts and guardrail confirmation

Forbidden outputs: labels, targets, predictions, ground-truth claims,
clustering-as-class, GIS-as-ground-truth, multimodal, heavy binaries in git.
All outputs go to local_runs/tcc_figures/v1gy/.
"""
from __future__ import annotations

import argparse
import csv as csv_module
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1gy"

DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "tcc_figures" / PHASE

V1GU_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gu"
V1GV_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gv"
V1GW_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gw"
V1GX_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gx"

METHODOLOGICAL_GUARDRAILS: dict[str, Any] = {
    "review_only": True,
    "no_labels": True,
    "no_targets": True,
    "no_predictions": True,
    "no_ground_truth_claim": True,
    "no_cluster_as_class": True,
    "gis_contextual_only": True,
    "multimodal_disabled": True,
    "no_heavy_files_in_git": True,
}

FORBIDDEN_CAPTION_TERMS = {
    "prediction", "detection", "ground truth", "ground-truth",
    "class", "label", "risk prediction",
}

REGION_COLORS = {
    "Curitiba": "#2196F3",
    "Petropolis": "#4CAF50",
    "Petrópolis": "#4CAF50",
    "Recife": "#FF9800",
}

REGION_DISPLAY = {
    "Curitiba": "Curitiba",
    "Petropolis": "Petrópolis",
    "Petrópolis": "Petrópolis",
    "Recife": "Recife",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv_module.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Caption validation
# ---------------------------------------------------------------------------

def check_caption(caption: str) -> list[str]:
    import re
    lower = caption.lower()
    return [
        term for term in FORBIDDEN_CAPTION_TERMS
        if re.search(r"\b" + re.escape(term) + r"\b", lower)
    ]


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def _patch_region(patch_id: str, reg_data: dict[str, Any]) -> str:
    for region, info in reg_data.get("centroids", {}).items():
        if patch_id in info.get("patches", []):
            return REGION_DISPLAY.get(region, region)
    return "Unknown"


def _region_color(region: str) -> str:
    for key, color in REGION_COLORS.items():
        if key.lower() in region.lower():
            return color
    return "#9E9E9E"


def generate_similarity_heatmap(
    sim_data: dict[str, Any],
    reg_data: dict[str, Any],
    output_path: Path,
) -> str:
    """12x12 cosine similarity heatmap. Returns status string."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return "BLOCKED_MATPLOTLIB_MISSING"

    patch_ids: list[str] = sim_data.get("patch_ids", [])
    matrix_dict: dict[str, dict[str, float]] = sim_data.get("matrix", {})
    if not patch_ids or not matrix_dict:
        return "BLOCKED_NO_DATA"

    n = len(patch_ids)
    mat = np.zeros((n, n), dtype=float)
    for i, pi in enumerate(patch_ids):
        row = matrix_dict.get(pi, [])
        for j, pj in enumerate(patch_ids):
            if isinstance(row, dict):
                mat[i, j] = float(row.get(pj, 0.0))
            elif isinstance(row, list) and j < len(row):
                mat[i, j] = float(row[j])

    regions = [_patch_region(pid, reg_data) for pid in patch_ids]
    colors = [_region_color(r) for r in regions]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, vmin=0.7, vmax=1.0, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [pid.replace("_00", "_") for pid in patch_ids]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)

    for i in range(n):
        for j in range(n):
            ax.add_patch(mpatches.FancyBboxPatch(
                (j - 0.5, i - 0.5), 1, 1,
                boxstyle="square,pad=0",
                fill=False,
                edgecolor=colors[i] if i == j else "none",
                linewidth=2,
            ))

    legend_handles = [
        mpatches.Patch(color=REGION_COLORS["Curitiba"], label="Curitiba"),
        mpatches.Patch(color=REGION_COLORS["Petropolis"], label="Petrópolis"),
        mpatches.Patch(color=REGION_COLORS["Recife"], label="Recife"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              title="Região", title_fontsize=8)

    ax.set_title(
        "Similaridade cosseno entre representações estruturais\n"
        "(DINOv2 com registros — análise exploratória, sem inferência)",
        fontsize=10, pad=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "READY"


def generate_neighbor_network(
    neighbors: list[dict[str, str]],
    reg_data: dict[str, Any],
    output_path: Path,
) -> str:
    """Nearest-neighbor graph by patch. Returns status string."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return "BLOCKED_MATPLOTLIB_MISSING"

    if not neighbors:
        return "BLOCKED_NO_DATA"

    patches = sorted({r["patch_id"] for r in neighbors})
    n = len(patches)
    idx = {pid: i for i, pid in enumerate(patches)}
    regions = {pid: _patch_region(pid, reg_data) for pid in patches}

    np.random.seed(42)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angle)
    y = np.sin(angle)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    drawn: set[tuple[int, int]] = set()
    for row in neighbors:
        src, tgt = row["patch_id"], row["neighbor_patch_id"]
        if src not in idx or tgt not in idx:
            continue
        i, j = idx[src], idx[tgt]
        key = (min(i, j), max(i, j))
        if key in drawn:
            continue
        drawn.add(key)
        same_region = regions[src] == regions[tgt]
        ax.plot(
            [x[i], x[j]], [y[i], y[j]],
            color="#BDBDBD" if not same_region else "#616161",
            linewidth=0.8 if not same_region else 1.5,
            alpha=0.6,
            zorder=1,
        )

    for pid in patches:
        i = idx[pid]
        color = _region_color(regions[pid])
        ax.scatter(x[i], y[i], color=color, s=120, zorder=3, edgecolors="white", linewidths=0.8)
        short = pid.replace("_00", "_")
        ax.text(x[i] * 1.12, y[i] * 1.12, short, ha="center", va="center", fontsize=6.5)

    legend_handles = [
        mpatches.Patch(color=REGION_COLORS["Curitiba"], label="Curitiba"),
        mpatches.Patch(color=REGION_COLORS["Petropolis"], label="Petrópolis"),
        mpatches.Patch(color=REGION_COLORS["Recife"], label="Recife"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              title="Região", title_fontsize=9)
    ax.set_title(
        "Rede de vizinhos mais próximos (top-5) por patch Sentinel\n"
        "(estrutura de similaridade exploratória — sem rótulos ou classes)",
        fontsize=10, pad=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "READY"


def generate_intra_inter_rate(
    reg_data: dict[str, Any],
    output_path: Path,
) -> str:
    """Bar chart of intra vs inter region neighbor rate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "BLOCKED_MATPLOTLIB_MISSING"

    analysis = reg_data.get("intra_inter_region_analysis", {})
    intra = analysis.get("intra_region_rate")
    inter = analysis.get("inter_region_rate")
    if intra is None or inter is None:
        return "BLOCKED_NO_DATA"

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Intra-região", "Inter-região"],
        [intra, inter],
        color=["#42A5F5", "#FFA726"],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, [intra, inter]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Taxa de vizinhos (top-5)", fontsize=10)
    ax.set_title(
        "Taxa de vizinhos intra e inter-região\n"
        f"(12 patches, {analysis.get('total_neighbor_edges', '?')} pares — análise exploratória)",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "READY"


def generate_review_category_figure(
    meta: dict[str, Any],
    output_path: Path,
) -> str:
    """Bar chart of review candidate categories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "BLOCKED_MATPLOTLIB_MISSING"

    category_counts: dict[str, int] = meta.get("category_counts", {})
    if not category_counts:
        return "BLOCKED_NO_DATA"

    labels = list(category_counts.keys())
    values = [category_counts[k] for k in labels]
    display_labels = [lbl.replace("_", "\n") for lbl in labels]

    colors = ["#5C6BC0", "#26A69A", "#FFA726", "#EF5350", "#AB47BC"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(display_labels, values,
                  color=colors[:len(labels)], edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Nº de patches candidatos", fontsize=10)
    ax.set_title(
        "Candidatos à revisão supervisora por categoria\n"
        f"(total: {meta.get('n_candidates', '?')} — EMBEDDING_BASED)",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "READY"


def generate_external_evidence_coverage(
    regional_summary: list[dict[str, str]],
    output_path: Path,
) -> str:
    """Stacked bar of coverage status per indicator per region."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "BLOCKED_MATPLOTLIB_MISSING"

    if not regional_summary:
        return "BLOCKED_NO_DATA"

    status_cols = ["AVAILABLE", "PARTIAL", "BBOX_ONLY", "BLOCKED",
                   "NOT_ACQUIRED", "LOCAL_ONLY", "MISSING"]
    status_colors = {
        "AVAILABLE": "#43A047",
        "PARTIAL": "#FFA726",
        "BBOX_ONLY": "#29B6F6",
        "BLOCKED": "#EF5350",
        "NOT_ACQUIRED": "#B0BEC5",
        "LOCAL_ONLY": "#AB47BC",
        "MISSING": "#ECEFF1",
    }

    regions = sorted({r["region"] for r in regional_summary})
    indicators = sorted({r["indicator_id"] for r in regional_summary})
    row_labels = [f"{r} / {ind}" for r in regions for ind in indicators
                  if any(s["region"] == r and s["indicator_id"] == ind
                         for s in regional_summary)]

    data_map = {(r["region"], r["indicator_id"]): r for r in regional_summary}

    row_keys = [(r, ind) for r in regions for ind in indicators
                if (r, ind) in data_map]
    row_labels = [f"{REGION_DISPLAY.get(r, r)} / {ind.replace('_', ' ')}"
                  for r, ind in row_keys]

    n_rows = len(row_keys)
    if n_rows == 0:
        return "BLOCKED_NO_DATA"

    fig, ax = plt.subplots(figsize=(9, max(4, n_rows * 0.45 + 1.5)))
    bottoms = np.zeros(n_rows)

    for status in status_cols:
        vals = []
        for r, ind in row_keys:
            row = data_map.get((r, ind), {})
            n_patches = int(row.get("n_patches", 1) or 1)
            count = int(row.get(status, 0) or 0)
            vals.append(count / n_patches * 100)
        bars = ax.barh(row_labels, vals, left=bottoms,
                       color=status_colors[status], label=status)
        bottoms += np.array(vals)

    ax.set_xlim(0, 105)
    ax.set_xlabel("% de patches por status", fontsize=9)
    ax.set_title(
        "Disponibilidade de evidência contextual externa por indicador e região\n"
        "(GIS como contexto territorial — sem inferência causal)",
        fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=7, title="Status", title_fontsize=7,
              ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "READY"


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def build_table_embedding_corpus_summary(reg_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    centroids = reg_data.get("centroids", {})
    medoids = reg_data.get("medoids_and_outliers", {})
    for region, info in centroids.items():
        med_info = medoids.get(region, {})
        rows.append({
            "regiao": REGION_DISPLAY.get(region, region),
            "n_patches": info.get("n_patches", ""),
            "norma_centroide": round(info.get("centroid_norm", 0), 4),
            "medoid": med_info.get("medoid", ""),
            "n_outliers": len(med_info.get("outliers", [])),
        })
    return rows


def build_table_medoids_outliers(reg_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for region, info in reg_data.get("medoids_and_outliers", {}).items():
        rows.append({
            "regiao": REGION_DISPLAY.get(region, region),
            "medoid": info.get("medoid", ""),
            "outliers": "; ".join(info.get("outliers", [])),
            "n_patches_regiao": info.get("n_patches", ""),
            "nota": "análise estrutural exploratória, sem rótulo de risco",
        })
    return rows


def build_table_neighbor_rate_summary(reg_data: dict[str, Any]) -> list[dict[str, Any]]:
    a = reg_data.get("intra_inter_region_analysis", {})
    if not a:
        return []
    return [{
        "total_pares_vizinhos": a.get("total_neighbor_edges", ""),
        "pares_intra_regiao": a.get("intra_region_edges", ""),
        "pares_inter_regiao": a.get("inter_region_edges", ""),
        "taxa_intra": round(a.get("intra_region_rate", 0), 4),
        "taxa_inter": round(a.get("inter_region_rate", 0), 4),
        "top_k": 5,
        "n_patches": 12,
        "nota": "inter > intra indica heterogeneidade estrutural entre regiões",
    }]


def build_table_review_candidates_summary(
    meta: dict[str, Any],
    candidates: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    category_counts = meta.get("category_counts", {})
    rows = []
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        rows.append({
            "categoria": cat,
            "n_candidatos": count,
            "modo_selecao": meta.get("embedding_mode", ""),
            "nota": "candidatos para revisão supervisora — sem classificação automática",
        })
    return rows


def build_table_external_evidence_summary(
    regional_summary: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for r in regional_summary:
        n = int(r.get("n_patches", 1) or 1)
        avail = int(r.get("AVAILABLE", 0) or 0) + int(r.get("PARTIAL", 0) or 0)
        rows.append({
            "regiao": REGION_DISPLAY.get(r["region"], r["region"]),
            "indicador": r.get("indicator_id", "").replace("_", " "),
            "n_patches": n,
            "disponivel_ou_parcial": avail,
            "nao_adquirido": int(r.get("NOT_ACQUIRED", 0) or 0),
            "ausente": int(r.get("MISSING", 0) or 0),
            "papel": "evidência contextual territorial — exclusivamente descritivo",
        })
    return rows


def build_table_figures_manifest(v1gx_dir: Path) -> list[dict[str, Any]]:
    figures_csv = v1gx_dir / "tcc_figures_export_plan_v1gx.csv"
    tables_csv = v1gx_dir / "tcc_tables_export_plan_v1gx.csv"
    rows = []
    for path in [figures_csv, tables_csv]:
        for r in read_csv(path):
            artifact_id = r.get("figure_id") or r.get("table_id", "")
            rows.append({
                "artifact_id": artifact_id,
                "titulo": r.get("title", ""),
                "secao_tcc": r.get("section", ""),
                "status": r.get("status", ""),
                "fonte": r.get("source_files", ""),
                "notas": r.get("notes", ""),
            })
    return rows


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(figure_statuses: dict[str, str]) -> list[dict[str, Any]]:
    entries = [
        {
            "artifact_id": "fig_similarity_heatmap_v1gy",
            "artifact_type": "figure",
            "filename": "fig_similarity_heatmap_v1gy.png",
            "status": figure_statuses.get("heatmap", "BLOCKED"),
            "source_stage": "v1gu",
            "source_file": "local_runs/dino_embeddings/v1gu/embedding_similarity_matrix_v1gu.json",
            "recommended_tcc_section": "4. Resultados | 4.1 Análise Estrutural dos Embeddings",
            "caption_draft_pt": (
                "Matriz de similaridade cosseno entre as representações estruturais "
                "dos 12 patches Sentinel analisados (DINOv2 com registros, 768 dimensões). "
                "Análise exploratória sem inferência preditiva."
            ),
            "claim_scope": "STRUCTURAL_EXPLORATION_ONLY",
            "limitations_note": (
                "Corpus reduzido (12 patches, 4 por região). "
                "Resultados não generalizam para o universo de 128 patches sem revisão supervisora."
            ),
        },
        {
            "artifact_id": "fig_neighbor_network_v1gy",
            "artifact_type": "figure",
            "filename": "fig_neighbor_network_v1gy.png",
            "status": figure_statuses.get("network", "BLOCKED"),
            "source_stage": "v1gu",
            "source_file": "local_runs/dino_embeddings/v1gu/embedding_neighbors_v1gu.csv",
            "recommended_tcc_section": "4. Resultados | 4.1 Análise Estrutural dos Embeddings",
            "caption_draft_pt": (
                "Rede de vizinhos mais próximos (top-5) entre patches Sentinel. "
                "Nós representam patches; arestas indicam alta similaridade estrutural. "
                "Cores por região geográfica. Análise exploratória."
            ),
            "claim_scope": "STRUCTURAL_EXPLORATION_ONLY",
            "limitations_note": (
                "Layout circular arbitrário (não espacial). "
                "Arestas inter-região dominantes indicam heterogeneidade — sem interpretação causal."
            ),
        },
        {
            "artifact_id": "fig_intra_inter_neighbor_rate_v1gy",
            "artifact_type": "figure",
            "filename": "fig_intra_inter_neighbor_rate_v1gy.png",
            "status": figure_statuses.get("rate", "BLOCKED"),
            "source_stage": "v1gu",
            "source_file": "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json",
            "recommended_tcc_section": "4. Resultados | 4.2 Estrutura Regional",
            "caption_draft_pt": (
                "Taxa de vizinhos mais próximos intra-região vs inter-região. "
                "Taxa inter-região de 63,3% sugere heterogeneidade estrutural entre regiões "
                "nas representações DINOv2. Análise exploratória com 12 patches (4 por região)."
            ),
            "claim_scope": "STRUCTURAL_EXPLORATION_ONLY",
            "limitations_note": (
                "Baseado em apenas 12 patches. "
                "Taxa inter > intra não implica ausência de padrão regional."
            ),
        },
        {
            "artifact_id": "fig_review_candidate_categories_v1gy",
            "artifact_type": "figure",
            "filename": "fig_review_candidate_categories_v1gy.png",
            "status": figure_statuses.get("categories", "BLOCKED"),
            "source_stage": "v1gw",
            "source_file": "local_runs/dino_embeddings/v1gw/review_candidates_metadata_v1gw.json",
            "recommended_tcc_section": "5. Metodologia | 5.1 Formalização da Etapa de Revisão",
            "caption_draft_pt": (
                "Distribuição de candidatos à revisão supervisora por categoria de seleção. "
                "Seleção baseada em métricas estruturais (EMBEDDING_BASED). "
                "Candidatos são indicados para revisão metodológica — sem rótulos automáticos."
            ),
            "claim_scope": "REVIEW_CANDIDATE_SELECTION_ONLY",
            "limitations_note": (
                "Categorias são critérios de seleção, não rótulos de risco ou vulnerabilidade. "
                "Revisão supervisora é etapa metodológica, não validação."
            ),
        },
        {
            "artifact_id": "fig_external_evidence_coverage_v1gy",
            "artifact_type": "figure",
            "filename": "fig_external_evidence_coverage_v1gy.png",
            "status": figure_statuses.get("coverage", "BLOCKED"),
            "source_stage": "v1gv",
            "source_file": "local_runs/dino_embeddings/v1gv/evidence_regional_summary_v1gv.csv",
            "recommended_tcc_section": "4. Resultados | 4.3 Disponibilidade de Evidência Contextual",
            "caption_draft_pt": (
                "Disponibilidade de evidência contextual externa por indicador e região. "
                "GIS utilizado exclusivamente como contexto territorial descritivo — "
                "sem função de validação ou comparação com embeddings."
            ),
            "claim_scope": "CONTEXTUAL_EVIDENCE_DOCUMENTATION_ONLY",
            "limitations_note": (
                "Status de cobertura reflete disponibilidade local de dados abertos. "
                "Indicadores ausentes documentados como NOT_ACQUIRED ou MISSING."
            ),
        },
        {
            "artifact_id": "table_embedding_corpus_summary_v1gy",
            "artifact_type": "table",
            "filename": "table_embedding_corpus_summary_v1gy.csv",
            "status": figure_statuses.get("table_corpus", "BLOCKED"),
            "source_stage": "v1gu",
            "source_file": "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json",
            "recommended_tcc_section": "4. Resultados | 4.1 Corpus de Embeddings",
            "caption_draft_pt": (
                "Estatísticas do corpus de embeddings por região. "
                "Norma do centroide regional e patches medoid identificados por análise estrutural."
            ),
            "claim_scope": "STRUCTURAL_EXPLORATION_ONLY",
            "limitations_note": "Corpus reduzido — 4 patches por região.",
        },
        {
            "artifact_id": "table_medoids_outliers_v1gy",
            "artifact_type": "table",
            "filename": "table_medoids_outliers_v1gy.csv",
            "status": figure_statuses.get("table_medoids", "BLOCKED"),
            "source_stage": "v1gu",
            "source_file": "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json",
            "recommended_tcc_section": "4. Resultados | 4.2 Estrutura Regional",
            "caption_draft_pt": (
                "Patches medoid e outliers estruturais por região. "
                "Medoid: patch mais central na distribuição de embeddings. "
                "Outlier: patch mais distante — priorizados para revisão supervisora."
            ),
            "claim_scope": "STRUCTURAL_EXPLORATION_ONLY",
            "limitations_note": (
                "Medoid e outlier são termos estruturais, não rótulos de risco. "
                "Revisão supervisora necessária para interpretação contextual."
            ),
        },
        {
            "artifact_id": "table_neighbor_rate_summary_v1gy",
            "artifact_type": "table",
            "filename": "table_neighbor_rate_summary_v1gy.csv",
            "status": figure_statuses.get("table_rate", "BLOCKED"),
            "source_stage": "v1gu",
            "source_file": "local_runs/dino_embeddings/v1gu/embedding_regional_summary_v1gu.json",
            "recommended_tcc_section": "4. Resultados | 4.2 Estrutura Regional",
            "caption_draft_pt": (
                "Resumo das taxas de vizinhança intra e inter-região. "
                "Análise exploratória com 12 patches e top-5 vizinhos."
            ),
            "claim_scope": "STRUCTURAL_EXPLORATION_ONLY",
            "limitations_note": "Corpus reduzido. Taxa inter > intra não é conclusiva.",
        },
        {
            "artifact_id": "table_review_candidates_summary_v1gy",
            "artifact_type": "table",
            "filename": "table_review_candidates_summary_v1gy.csv",
            "status": figure_statuses.get("table_review", "BLOCKED"),
            "source_stage": "v1gw",
            "source_file": "local_runs/dino_embeddings/v1gw/review_candidates_metadata_v1gw.json",
            "recommended_tcc_section": "5. Metodologia | 5.1 Formalização da Etapa de Revisão",
            "caption_draft_pt": (
                "Distribuição de candidatos à revisão supervisora por categoria. "
                "Candidatos selecionados por métricas estruturais — sem atribuição automática "
                "de risco ou vulnerabilidade."
            ),
            "claim_scope": "REVIEW_CANDIDATE_SELECTION_ONLY",
            "limitations_note": "Candidatos requerem revisão supervisora antes de qualquer interpretação.",
        },
        {
            "artifact_id": "table_external_evidence_coverage_summary_v1gy",
            "artifact_type": "table",
            "filename": "table_external_evidence_coverage_summary_v1gy.csv",
            "status": figure_statuses.get("table_coverage", "BLOCKED"),
            "source_stage": "v1gv",
            "source_file": "local_runs/dino_embeddings/v1gv/evidence_regional_summary_v1gv.csv",
            "recommended_tcc_section": "4. Resultados | 4.3 Evidência Contextual Externa",
            "caption_draft_pt": (
                "Resumo de disponibilidade de evidência contextual GIS por indicador e região. "
                "GIS como contexto territorial, não como validação."
            ),
            "claim_scope": "CONTEXTUAL_EVIDENCE_DOCUMENTATION_ONLY",
            "limitations_note": "Cobertura parcial em todas as regiões.",
        },
        {
            "artifact_id": "table_figures_for_tcc_manifest_v1gy",
            "artifact_type": "table",
            "filename": "table_figures_for_tcc_manifest_v1gy.csv",
            "status": figure_statuses.get("table_plan", "BLOCKED"),
            "source_stage": "v1gx",
            "source_file": "local_runs/dino_embeddings/v1gx/tcc_figures_export_plan_v1gx.csv",
            "recommended_tcc_section": "Apêndice",
            "caption_draft_pt": (
                "Plano de figuras e tabelas TCC: status de prontidão por artefato."
            ),
            "claim_scope": "DOCUMENTATION_ONLY",
            "limitations_note": "Referência interna para escrita do TCC.",
        },
    ]

    for entry in entries:
        violations = check_caption(entry.get("caption_draft_pt", ""))
        entry["caption_forbidden_terms"] = "; ".join(violations) if violations else "none"

    return entries


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gy: TCC visual evidence export package."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--v1gu-dir", default=str(V1GU_DIR),
        help="Path to v1gu outputs directory",
    )
    parser.add_argument(
        "--v1gv-dir", default=str(V1GV_DIR),
        help="Path to v1gv outputs directory",
    )
    parser.add_argument(
        "--v1gw-dir", default=str(V1GW_DIR),
        help="Path to v1gw outputs directory",
    )
    parser.add_argument(
        "--v1gx-dir", default=str(V1GX_DIR),
        help="Path to v1gx outputs directory",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    v1gu_dir = Path(args.v1gu_dir)
    v1gv_dir = Path(args.v1gv_dir)
    v1gw_dir = Path(args.v1gw_dir)
    v1gx_dir = Path(args.v1gx_dir)

    if output_dir.exists() and not args.force:
        print(f"[v1gy] Output directory already exists: {output_dir}. Use --force.", file=sys.stderr)
        return 1
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v1gy] Output: {rel(output_dir)}")

    # --- load inputs ---
    sim_data = load_json(v1gu_dir / "embedding_similarity_matrix_v1gu.json")
    reg_data = load_json(v1gu_dir / "embedding_regional_summary_v1gu.json")
    neighbors = read_csv(v1gu_dir / "embedding_neighbors_v1gu.csv")
    regional_summary = read_csv(v1gv_dir / "evidence_regional_summary_v1gv.csv")
    candidates = read_csv(v1gw_dir / "review_candidates_v1gw.csv")
    v1gw_meta = load_json(v1gw_dir / "review_candidates_metadata_v1gw.json")

    sources_found = []
    sources_missing = []
    for label, path in [
        ("v1gu/embedding_similarity_matrix", v1gu_dir / "embedding_similarity_matrix_v1gu.json"),
        ("v1gu/embedding_regional_summary", v1gu_dir / "embedding_regional_summary_v1gu.json"),
        ("v1gu/embedding_neighbors", v1gu_dir / "embedding_neighbors_v1gu.csv"),
        ("v1gv/evidence_regional_summary", v1gv_dir / "evidence_regional_summary_v1gv.csv"),
        ("v1gw/review_candidates", v1gw_dir / "review_candidates_v1gw.csv"),
        ("v1gw/review_candidates_metadata", v1gw_dir / "review_candidates_metadata_v1gw.json"),
        ("v1gx/tcc_figures_export_plan", v1gx_dir / "tcc_figures_export_plan_v1gx.csv"),
    ]:
        (sources_found if path.exists() else sources_missing).append(label)

    figure_statuses: dict[str, str] = {}

    # --- figures ---
    print("[v1gy] Generating figures...")

    status = generate_similarity_heatmap(
        sim_data, reg_data, output_dir / "fig_similarity_heatmap_v1gy.png"
    )
    figure_statuses["heatmap"] = status
    print(f"  heatmap: {status}")

    status = generate_neighbor_network(
        neighbors, reg_data, output_dir / "fig_neighbor_network_v1gy.png"
    )
    figure_statuses["network"] = status
    print(f"  network: {status}")

    status = generate_intra_inter_rate(
        reg_data, output_dir / "fig_intra_inter_neighbor_rate_v1gy.png"
    )
    figure_statuses["rate"] = status
    print(f"  intra/inter rate: {status}")

    status = generate_review_category_figure(
        v1gw_meta, output_dir / "fig_review_candidate_categories_v1gy.png"
    )
    figure_statuses["categories"] = status
    print(f"  review categories: {status}")

    status = generate_external_evidence_coverage(
        regional_summary, output_dir / "fig_external_evidence_coverage_v1gy.png"
    )
    figure_statuses["coverage"] = status
    print(f"  external coverage: {status}")

    # --- tables ---
    print("[v1gy] Generating tables...")

    tbl_corpus = build_table_embedding_corpus_summary(reg_data)
    if tbl_corpus:
        write_csv(
            output_dir / "table_embedding_corpus_summary_v1gy.csv",
            tbl_corpus,
            ["regiao", "n_patches", "norma_centroide", "medoid", "n_outliers"],
        )
        figure_statuses["table_corpus"] = "READY"
    else:
        figure_statuses["table_corpus"] = "BLOCKED_NO_DATA"
    print(f"  table_corpus: {figure_statuses['table_corpus']}")

    tbl_medoids = build_table_medoids_outliers(reg_data)
    if tbl_medoids:
        write_csv(
            output_dir / "table_medoids_outliers_v1gy.csv",
            tbl_medoids,
            ["regiao", "medoid", "outliers", "n_patches_regiao", "nota"],
        )
        figure_statuses["table_medoids"] = "READY"
    else:
        figure_statuses["table_medoids"] = "BLOCKED_NO_DATA"
    print(f"  table_medoids: {figure_statuses['table_medoids']}")

    tbl_rate = build_table_neighbor_rate_summary(reg_data)
    if tbl_rate:
        write_csv(
            output_dir / "table_neighbor_rate_summary_v1gy.csv",
            tbl_rate,
            ["total_pares_vizinhos", "pares_intra_regiao", "pares_inter_regiao",
             "taxa_intra", "taxa_inter", "top_k", "n_patches", "nota"],
        )
        figure_statuses["table_rate"] = "READY"
    else:
        figure_statuses["table_rate"] = "BLOCKED_NO_DATA"
    print(f"  table_rate: {figure_statuses['table_rate']}")

    tbl_review = build_table_review_candidates_summary(v1gw_meta, candidates)
    if tbl_review:
        write_csv(
            output_dir / "table_review_candidates_summary_v1gy.csv",
            tbl_review,
            ["categoria", "n_candidatos", "modo_selecao", "nota"],
        )
        figure_statuses["table_review"] = "READY"
    else:
        figure_statuses["table_review"] = "BLOCKED_NO_DATA"
    print(f"  table_review: {figure_statuses['table_review']}")

    tbl_coverage = build_table_external_evidence_summary(regional_summary)
    if tbl_coverage:
        write_csv(
            output_dir / "table_external_evidence_coverage_summary_v1gy.csv",
            tbl_coverage,
            ["regiao", "indicador", "n_patches", "disponivel_ou_parcial",
             "nao_adquirido", "ausente", "papel"],
        )
        figure_statuses["table_coverage"] = "READY"
    else:
        figure_statuses["table_coverage"] = "BLOCKED_NO_DATA"
    print(f"  table_coverage: {figure_statuses['table_coverage']}")

    tbl_plan = build_table_figures_manifest(v1gx_dir)
    if tbl_plan:
        write_csv(
            output_dir / "table_figures_for_tcc_manifest_v1gy.csv",
            tbl_plan,
            ["artifact_id", "titulo", "secao_tcc", "status", "fonte", "notas"],
        )
        figure_statuses["table_plan"] = "READY"
    else:
        figure_statuses["table_plan"] = "BLOCKED_NO_DATA"
    print(f"  table_plan: {figure_statuses['table_plan']}")

    # --- manifest ---
    print("[v1gy] Writing manifest...")
    manifest = build_manifest(figure_statuses)
    write_csv(
        output_dir / "tcc_visual_evidence_manifest_v1gy.csv",
        manifest,
        [
            "artifact_id", "artifact_type", "filename", "status",
            "source_stage", "source_file", "recommended_tcc_section",
            "caption_draft_pt", "claim_scope", "limitations_note",
            "caption_forbidden_terms",
        ],
    )

    # --- summary ---
    figures_ready = sum(1 for k, v in figure_statuses.items()
                        if not k.startswith("table") and v == "READY")
    figures_blocked = sum(1 for k, v in figure_statuses.items()
                          if not k.startswith("table") and v != "READY")
    tables_ready = sum(1 for k, v in figure_statuses.items()
                       if k.startswith("table") and v == "READY")

    summary: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "total_figures_ready": figures_ready,
        "total_figures_blocked": figures_blocked,
        "total_tables_ready": tables_ready,
        "figure_statuses": figure_statuses,
        "source_inputs_found": sources_found,
        "source_inputs_missing": sources_missing,
        "forbidden_claims_checked": True,
        "no_ground_truth_claim": True,
        "no_prediction_claim": True,
        "methodological_guardrails": METHODOLOGICAL_GUARDRAILS,
    }
    write_json(output_dir / "tcc_visual_evidence_summary_v1gy.json", summary)

    print(f"[v1gy] Done — figures: {figures_ready} READY / {figures_blocked} BLOCKED | tables: {tables_ready} READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
