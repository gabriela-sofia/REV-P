"""REV-P v1qz — DINO embeddings joined to real SEDEC Recife records (descriptive only).

Spatial-matches the 163-record primary real-vs-real SEDEC dataset (from
`recife_modelo_final_v5/pipeline_final_v5.py`, PROJETO) to the 37 real Recife
Sentinel patches by point-in-bbox containment (SEDEC lat/lon vs. patch raster
bounds reprojected to WGS84). Only 14/163 records fall inside a patch we have
Sentinel imagery for (8 unique patches) — this script reports that honestly
and computes a DESCRIPTIVE-ONLY comparison of DINO embedding structure
(PCA projection) between label=1 (n=11) and label=0 (n=3).

Methodological boundary (deliberately strict here, stricter than v1qx):
  * n=3 negatives is far below the >=10 events-per-predictor heuristic that
    the underlying pipeline_final_v5.py itself already flagged as fragile at
    n=22. NO p-value, AUC, or coefficient from this script may be read as
    evidence of anything. Output is descriptive statistics only.
  * This does NOT retrain or extend the Firth/bootstrap/LOO-AUC model in
    pipeline_final_v5.py. It is a separate, clearly-labeled diagnostic.
  * Never creates a label, target, or ground truth. DINO embeddings remain
    auxiliary/visual evidence, never causal.
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

IN_EMBEDDINGS = _p("REVP_V1QZ_IN_EMB", DATASETS / "dino_recife_sedec_all_embeddings_v1r3.csv")
IN_SEDEC = _p(
    "REVP_V1QZ_IN_SEDEC",
    Path("/sessions/affectionate-optimistic-davinci/mnt/PROJETO/local_runs/"
         "recife_modelo_oficial_v4_auditoria_lacunas/dataset_v4_features_finais.csv"),
)
OUT_JOIN = _p("REVP_V1QZ_OUT_JOIN", DATASETS / "dino_sedec_recife_join_v1qz.csv")
OUT_SUM = _p("REVP_V1QZ_OUT_SUM", DATASETS / "dino_sedec_recife_join_summary_v1qz.csv")
SCH_JOIN = _p("REVP_V1QZ_SCH_JOIN", SCHEMAS / "dino_sedec_recife_join_v1qz_schema.csv")
DOC = _p("REVP_V1QZ_DOC", DOCS / "revp_v1qz_dino_sedec_recife_join_descriptive.md")

PRIMARY_TYPES = {"flood_positive", "real_clean_sedec_negative"}

JOIN_FIELDS = [
    "point_id", "label", "negative_source_type", "matched_patch_id",
    "pca_x", "pca_y", "representation_use",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _patch_bboxes_wgs84() -> dict[str, tuple[float, float, float, float]]:
    import glob
    import rasterio
    from rasterio.warp import transform_bounds

    sentinel_dir = Path("/sessions/affectionate-optimistic-davinci/mnt/PROJETO/data/sentinel")
    out: dict[str, tuple[float, float, float, float]] = {}
    for p in sorted(glob.glob(str(sentinel_dir / "patch_recife_*.tif"))):
        with rasterio.open(p) as ds:
            b = ds.bounds
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
        pid = Path(p).stem.split("patch_recife_")[-1].upper()
        out[pid] = (lon_min, lat_min, lon_max, lat_max)
    return out


def match_sedec_to_patches() -> list[dict[str, str]]:
    rows = read_csv(IN_SEDEC)
    bboxes = _patch_bboxes_wgs84()
    matches: list[dict[str, str]] = []
    for r in rows:
        if r.get("negative_source_type", "") not in PRIMARY_TYPES:
            continue
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
        except (KeyError, ValueError):
            continue
        for pid, (xmin, ymin, xmax, ymax) in bboxes.items():
            if xmin <= lon <= xmax and ymin <= lat <= ymax:
                matches.append({
                    "point_id": r["point_id"], "label": r["label"],
                    "negative_source_type": r["negative_source_type"],
                    "matched_patch_id": f"REC_{pid}",
                })
                break
    return matches


def run() -> None:
    matches = match_sedec_to_patches()
    emb_rows = read_csv(IN_EMBEDDINGS)
    emb_by_patch = {r["patch_id"]: parse_embedding_from_row(r) for r in emb_rows}

    joined: list[dict[str, Any]] = []
    for m in matches:
        vec = emb_by_patch.get(m["matched_patch_id"])
        if vec is None:
            continue
        joined.append({**m, "vector": vec})

    n = len(joined)
    n_pos = sum(1 for j in joined if j["label"] == "1")
    n_neg = n - n_pos
    unique_patches = sorted({j["matched_patch_id"] for j in joined})

    if n < 4:
        write_csv(OUT_JOIN, [], JOIN_FIELDS)
        write_csv(OUT_SUM, [
            {"stat_key": "n_joined", "stat_value": str(n)},
            {"stat_key": "status", "stat_value": "JOIN_FAIL_CLOSED_N_LT_4"},
        ], SUM_FIELDS)
        print(f"[v1qz] n={n} status=FAIL_CLOSED_N_LT_4")
        return

    from sklearn.decomposition import PCA
    import numpy as np
    x = np.asarray([j["vector"] for j in joined], dtype=float)
    proj = PCA(n_components=2, svd_solver="full", random_state=0).fit_transform(x)

    join_out: list[dict[str, Any]] = []
    pca1_pos, pca1_neg, pca2_pos, pca2_neg = [], [], [], []
    for j, (px, py) in zip(joined, proj):
        join_out.append({
            "point_id": j["point_id"], "label": j["label"],
            "negative_source_type": j["negative_source_type"],
            "matched_patch_id": j["matched_patch_id"],
            "pca_x": f"{float(px):.6f}", "pca_y": f"{float(py):.6f}",
            "representation_use": "DESCRIPTIVE_ONLY_UNDERPOWERED_N",
        })
        (pca1_pos if j["label"] == "1" else pca1_neg).append(float(px))
        (pca2_pos if j["label"] == "1" else pca2_neg).append(float(py))

    write_csv(OUT_JOIN, join_out, JOIN_FIELDS)
    write_schema(SCH_JOIN, JOIN_FIELDS, "v1qz_dino_sedec_recife_join")

    def _mean(v: list[float]) -> float:
        return sum(v) / len(v) if v else float("nan")

    summary = [
        {"stat_key": "sedec_primary_pool_n", "stat_value": "163"},
        {"stat_key": "n_matched_to_sentinel_patch", "stat_value": str(len(matches))},
        {"stat_key": "n_with_real_dino_embedding", "stat_value": str(n)},
        {"stat_key": "n_pos_flood_positive", "stat_value": str(n_pos)},
        {"stat_key": "n_neg_real_clean_sedec_negative", "stat_value": str(n_neg)},
        {"stat_key": "n_unique_patches", "stat_value": str(len(unique_patches))},
        {"stat_key": "unique_patches", "stat_value": "|".join(unique_patches)},
        {"stat_key": "pca1_mean_label1", "stat_value": f"{_mean(pca1_pos):.4f}"},
        {"stat_key": "pca1_mean_label0", "stat_value": f"{_mean(pca1_neg):.4f}"},
        {"stat_key": "pca2_mean_label1", "stat_value": f"{_mean(pca2_pos):.4f}"},
        {"stat_key": "pca2_mean_label0", "stat_value": f"{_mean(pca2_neg):.4f}"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "used_as_training_feature", "stat_value": "false"},
        {"stat_key": "statistical_validity_caveat",
         "stat_value": f"n_neg={n_neg}_is_far_below_any_usable_threshold_descriptive_only_no_inference"},
        {"stat_key": "status", "stat_value": "DESCRIPTIVE_ONLY_REVIEW_READY"},
    ]
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_doc(DOC, "v1qz — DINO x SEDEC Recife Join (descriptive only, not a model)", [
        "## Objetivo",
        "Verificar honestamente quantos dos 163 registros SEDEC reais (primary "
        "real-vs-real do pipeline_final_v5.py) caem dentro dos 37 patches Sentinel "
        "reais que já têm embedding DINO, e reportar uma comparação puramente "
        "descritiva — nunca inferencial — da estrutura do embedding entre label=1 e "
        "label=0.",
        "## Por que não é um modelo",
        f"Apenas {len(matches)}/163 registros caem dentro de algum patch com Sentinel "
        f"real (n={n} com embedding válido; {len(unique_patches)} patches únicos). Com "
        f"{n_neg} negativos, qualquer p-valor, AUC ou coeficiente seria ruído, não "
        "evidência — muito abaixo do que o próprio pipeline_final_v5.py já sinalizou "
        "como frágil em n=22 negativos. Nenhum número aqui deve ser citado como achado.",
        "## Caminho para crescer isso de verdade",
        "Só aumentando a cobertura de patches Sentinel reais de Recife (hoje 37 de "
        "~100+ do dataset_final.csv) via exportação GEE adicional é que esse n cresce "
        "de forma honesta — não há atalho estatístico para contornar a falta de dado.",
    ])
    print(f"[v1qz] matched={len(matches)} with_embedding={n} pos={n_pos} neg={n_neg} "
          f"unique_patches={len(unique_patches)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qz dino sedec recife join descriptive").parse_args()
    run()
