"""REV-P v1qx — DINO embedding vs real physical-variable correlation (review-only).

First concrete step toward a multimodal framework: checks whether the DINO
visual embedding space correlates with REAL, independently measured physical
variables (elevation, slope) for the subset of Petropolis patches where both
exist (n=17: real Sentinel imagery in the DINO corpus AND real local
SGB/CPRM MDE + IBGE BC25 terrain evidence in
`petropolis_patch_terrain_context_features_v1.csv`).

Methodological boundary (never crossed by this script):
  * This produces a CORRELATION REPORT, not a model feature. No column here
    is ever written back as an input to a classifier.
  * DINO/orbital embeddings remain auxiliary evidence; elevation/slope are the
    causal physical variables. This script tests whether the auxiliary
    representation is *consistent with* the causal variables — it never
    treats the correlation itself as proof of anything, and it never creates
    a label, target, or ground truth.
  * n=17 is small. All correlations here are exploratory/directional only,
    reported with sample size, never with significance-cutoff language.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, parse_embedding_from_row, read_csv, write_csv, write_doc, write_schema,
)

IN_EMBEDDINGS = _p("REVP_V1QX_IN_EMB", DATASETS / "dino_petropolis_terrain_overlap_embeddings_v1qw.csv")
IN_TERRAIN = _p(
    "REVP_V1QX_IN_TERRAIN",
    Path("/sessions/affectionate-optimistic-davinci/mnt/PROJETO/outputs/external_validation/"
         "petropolis_sidecar/petropolis_patch_terrain_context_features_v1.csv"),
)
OUT_PAIRWISE = _p("REVP_V1QX_OUT_PAIRWISE", DATASETS / "dino_physical_correlation_pairwise_v1qx.csv")
OUT_PCA = _p("REVP_V1QX_OUT_PCA", DATASETS / "dino_physical_correlation_pca_v1qx.csv")
OUT_SUM = _p("REVP_V1QX_OUT_SUM", DATASETS / "dino_physical_correlation_summary_v1qx.csv")
SCH_PAIRWISE = _p("REVP_V1QX_SCH_PAIRWISE", SCHEMAS / "dino_physical_correlation_pairwise_v1qx_schema.csv")
DOC = _p("REVP_V1QX_DOC", DOCS / "revp_v1qx_dino_embedding_physical_variable_correlation.md")

PAIRWISE_FIELDS = [
    "patch_id_a", "patch_id_b", "embedding_cosine_similarity", "embedding_euclidean_distance",
    "abs_delta_elevation_mean", "abs_delta_slope_mean", "representation_use",
]
PCA_FIELDS = [
    "patch_id", "pca_x", "pca_y", "elevation_mean", "slope_mean",
    "mde_range", "ibge_hydro_features", "representation_use",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else float("nan")


def _euclid(a: list[float], b: list[float]) -> float:
    import math
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _pearson(xs: list[float], ys: list[float]) -> float:
    import numpy as np
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.std() == 0 or y.std() == 0 or len(x) < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_terrain(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        pid = (r.get("patch_id") or "").strip()
        if pid:
            out[pid] = r
    return out


def run() -> None:
    emb_rows = read_csv(IN_EMBEDDINGS)
    terrain = load_terrain(IN_TERRAIN)

    joined: list[dict[str, Any]] = []
    for r in emb_rows:
        patch_id = r.get("patch_id", "")  # e.g. PET_00291
        num = patch_id.split("_")[-1]
        terrain_key = f"petropolis_{num}"
        t = terrain.get(terrain_key)
        if t is None:
            continue
        vec = parse_embedding_from_row(r)
        if vec is None:
            continue
        try:
            elevation_mean = float(t.get("elevation_mean", "") or "nan")
            slope_mean = float(t.get("slope_mean", "") or "nan")
            mde_range = float(t.get("elevation_range", "") or "nan")
        except ValueError:
            continue
        joined.append({
            "patch_id": patch_id, "vector": vec,
            "elevation_mean": elevation_mean, "slope_mean": slope_mean,
            "mde_range": mde_range,
            "ibge_hydro_features": t.get("ibge_hydro_features", ""),
        })

    n = len(joined)
    if n < 3:
        write_csv(OUT_PAIRWISE, [], PAIRWISE_FIELDS)
        write_csv(OUT_PCA, [], PCA_FIELDS)
        write_csv(OUT_SUM, [
            {"stat_key": "n_joined_patches", "stat_value": str(n)},
            {"stat_key": "status", "stat_value": "CORRELATION_FAIL_CLOSED_N_LT_3"},
        ], SUM_FIELDS)
        print(f"[v1qx] n={n} status=FAIL_CLOSED_N_LT_3")
        return

    # --- Pairwise: embedding distance vs physical-variable distance ---
    pairwise: list[dict[str, Any]] = []
    emb_cos, delta_elev, delta_slope = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = joined[i], joined[j]
            cos = _cosine(a["vector"], b["vector"])
            eucl = _euclid(a["vector"], b["vector"])
            d_elev = abs(a["elevation_mean"] - b["elevation_mean"])
            d_slope = abs(a["slope_mean"] - b["slope_mean"])
            pairwise.append({
                "patch_id_a": a["patch_id"], "patch_id_b": b["patch_id"],
                "embedding_cosine_similarity": f"{cos:.6f}",
                "embedding_euclidean_distance": f"{eucl:.6f}",
                "abs_delta_elevation_mean": f"{d_elev:.4f}",
                "abs_delta_slope_mean": f"{d_slope:.4f}",
                "representation_use": "EXPLORATORY_CORRELATION_ONLY",
            })
            emb_cos.append(cos); delta_elev.append(d_elev); delta_slope.append(d_slope)

    # Correlation: does embedding SIMILARITY drop as physical-variable DISTANCE grows?
    # (negative r is the "expected" direction if embeddings track physical structure)
    r_elev = _pearson(emb_cos, delta_elev)
    r_slope = _pearson(emb_cos, delta_slope)

    # --- Direct: PCA(2D) of the 17 embeddings vs elevation/slope ---
    vectors = [j["vector"] for j in joined]
    try:
        from sklearn.decomposition import PCA
        import numpy as np
        x = np.asarray(vectors, dtype=float)
        proj = PCA(n_components=2, svd_solver="full", random_state=0).fit_transform(x)
    except Exception:
        proj = [[0.0, 0.0] for _ in vectors]

    pca_rows: list[dict[str, Any]] = []
    pca_x, pca_y, elevs, slopes = [], [], [], []
    for j, (px, py) in zip(joined, proj):
        pca_rows.append({
            "patch_id": j["patch_id"], "pca_x": f"{float(px):.6f}", "pca_y": f"{float(py):.6f}",
            "elevation_mean": f"{j['elevation_mean']:.4f}", "slope_mean": f"{j['slope_mean']:.4f}",
            "mde_range": f"{j['mde_range']:.4f}", "ibge_hydro_features": j["ibge_hydro_features"],
            "representation_use": "EXPLORATORY_CORRELATION_ONLY",
        })
        pca_x.append(float(px)); pca_y.append(float(py))
        elevs.append(j["elevation_mean"]); slopes.append(j["slope_mean"])

    r_pcax_elev = _pearson(pca_x, elevs)
    r_pcay_elev = _pearson(pca_y, elevs)
    r_pcax_slope = _pearson(pca_x, slopes)
    r_pcay_slope = _pearson(pca_y, slopes)

    write_csv(OUT_PAIRWISE, pairwise, PAIRWISE_FIELDS)
    write_csv(OUT_PCA, pca_rows, PCA_FIELDS)
    write_schema(SCH_PAIRWISE, PAIRWISE_FIELDS, "v1qx_dino_physical_correlation_pairwise")

    summary = [
        {"stat_key": "n_joined_patches", "stat_value": str(n)},
        {"stat_key": "n_pairs", "stat_value": str(len(pairwise))},
        {"stat_key": "pearson_r_embcos_vs_abs_delta_elevation", "stat_value": f"{r_elev:.4f}"},
        {"stat_key": "pearson_r_embcos_vs_abs_delta_slope", "stat_value": f"{r_slope:.4f}"},
        {"stat_key": "pearson_r_pca1_vs_elevation_mean", "stat_value": f"{r_pcax_elev:.4f}"},
        {"stat_key": "pearson_r_pca2_vs_elevation_mean", "stat_value": f"{r_pcay_elev:.4f}"},
        {"stat_key": "pearson_r_pca1_vs_slope_mean", "stat_value": f"{r_pcax_slope:.4f}"},
        {"stat_key": "pearson_r_pca2_vs_slope_mean", "stat_value": f"{r_pcay_slope:.4f}"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "used_as_training_feature", "stat_value": "false"},
        {"stat_key": "sample_size_caveat", "stat_value": f"n={n}_exploratory_only_not_statistically_robust"},
        {"stat_key": "status", "stat_value": "CORRELATION_READY_REVIEW_ONLY"},
    ]
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_doc(DOC, "v1qx — DINO Embedding vs Real Physical-Variable Correlation (review-only)", [
        "## Objetivo",
        "Primeiro passo concreto rumo a um framework multimodal: testar se o espaço de "
        "embedding DINO (auxiliar) é consistente com variáveis físicas reais e "
        "independentes (elevação, declividade — causais), para os "
        f"{n} patches de Petrópolis onde ambas existem.",
        "## Limite metodológico",
        "Isto é um relatório de correlação exploratória, nunca uma feature de modelo. "
        "Nenhuma coluna aqui é usada como entrada de classificador. n pequeno "
        "(exploratório, não robusto estatisticamente).",
        "## Resultado",
        f"r(similaridade_embedding, |Δelevação|) = {r_elev:.4f}; "
        f"r(similaridade_embedding, |Δdeclividade|) = {r_slope:.4f}; "
        f"r(PCA1, elevação) = {r_pcax_elev:.4f}; r(PCA1, declividade) = {r_pcax_slope:.4f}.",
    ])
    print(f"[v1qx] n={n} pairs={len(pairwise)} "
          f"r_cos_vs_delta_elev={r_elev:.4f} r_cos_vs_delta_slope={r_slope:.4f} "
          f"r_pca1_elev={r_pcax_elev:.4f} r_pca1_slope={r_pcax_slope:.4f}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qx dino vs physical variable correlation").parse_args()
    run()
