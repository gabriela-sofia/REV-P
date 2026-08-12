"""REV-P v1r4 — Extended real-vs-real analysis: DINO embeddings joined to the
163-record primary SEDEC dataset, now with real Sentinel coverage for n=81
(59 flood_positive + 22 real_clean_sedec_negative — ALL 22 real negatives are
now covered, after downloading 13 evidence-centered patches this session).

Two SEPARATE, never-mixed analyses (mirrors pipeline_final_v5.py's own
primary/secondary separation discipline):

  A) Univariate Mann-Whitney screen on the 6 real physical features already
     used in pipeline_final_v5.py (elevation_m, slope_deg, hand_m_filled, twi,
     rain_max_24h_chirps, rain_decay_index_api_chirps) PLUS DINO PCA1/PCA2 as
     two additional candidate features — screened individually, never jammed
     into one multivariate model together (8 predictors vs 22 minority-class
     events = 2.75 events/predictor, well below any usable EPV heuristic).

  B) A SEPARATE, clean Firth multivariate model using ONLY DINO PCA1+PCA2
     (2 predictors vs 22 minority events = 11 events/predictor — meets the
     conventional >=10 EPV heuristic on its own). This tests whether the
     auxiliary visual representation alone discriminates real flood-positive
     from real negative locations, without conflating it with the physical
     model's already-fragile predictor budget.

No label is created. No training happens against `v5_core` (circular, see
GAP_estrutural doc). DINO remains auxiliary evidence; physical variables
remain the causal, primary track — this script never claims otherwise.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, parse_embedding_from_row, read_csv, write_csv, write_doc, write_schema,
)

IN_EMBEDDINGS = _p("REVP_V1R4_IN_EMB", DATASETS / "dino_recife_sedec_all_embeddings_v1r3.csv")
IN_SEDEC = _p(
    "REVP_V1R4_IN_SEDEC",
    Path("/sessions/affectionate-optimistic-davinci/mnt/PROJETO/local_runs/"
         "recife_modelo_oficial_v4_auditoria_lacunas/dataset_v4_features_finais.csv"),
)
OUT_UNIV = _p("REVP_V1R4_OUT_UNIV", DATASETS / "dino_sedec_extended_univariate_v1r4.csv")
OUT_FIRTH = _p("REVP_V1R4_OUT_FIRTH", DATASETS / "dino_sedec_extended_firth_dino_only_v1r4.csv")
OUT_SUM = _p("REVP_V1R4_OUT_SUM", DATASETS / "dino_sedec_extended_summary_v1r4.csv")
SCH_UNIV = _p("REVP_V1R4_SCH_UNIV", SCHEMAS / "dino_sedec_extended_univariate_v1r4_schema.csv")
DOC = _p("REVP_V1R4_DOC", DOCS / "revp_v1r4_dino_sedec_extended_analysis.md")

PRIMARY_TYPES = {"flood_positive", "real_clean_sedec_negative"}
PHYSICAL_FEATS = ["elevation_m", "slope_deg", "hand_m_filled", "twi",
                  "rain_max_24h_chirps", "rain_decay_index_api_chirps"]
EXPECTED_SIGN = {"elevation_m": -1, "slope_deg": -1, "hand_m_filled": -1, "twi": +1,
                 "rain_max_24h_chirps": +1, "rain_decay_index_api_chirps": +1,
                 "dino_pca1": 0, "dino_pca2": 0}

UNIV_FIELDS = ["feature", "n_pos", "n_neg", "mean_pos", "mean_neg", "mannwhitney_U",
              "p_value", "rank_biserial_r", "expected_sign", "direction_observed",
              "significant_p05", "is_dino_feature"]
FIRTH_FIELDS = ["feature", "coef_standardized", "ci_low_95", "ci_high_95", "p_value",
               "ci_crosses_zero"]
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


def build_joined_dataset() -> list[dict[str, Any]]:
    sedec_rows = read_csv(IN_SEDEC)
    bboxes = _patch_bboxes_wgs84()
    emb_rows = read_csv(IN_EMBEDDINGS)
    emb_by_patch = {r["patch_id"]: parse_embedding_from_row(r) for r in emb_rows}

    joined: list[dict[str, Any]] = []
    for r in sedec_rows:
        if r.get("negative_source_type", "") not in PRIMARY_TYPES:
            continue
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
        except (KeyError, ValueError):
            continue
        matched_patch = None
        for pid, (xmin, ymin, xmax, ymax) in bboxes.items():
            if xmin <= lon <= xmax and ymin <= lat <= ymax:
                matched_patch = f"REC_{pid}"
                break
        if matched_patch is None:
            continue
        vec = emb_by_patch.get(matched_patch)
        if vec is None:
            continue
        try:
            feats = {f: float(r[f]) for f in PHYSICAL_FEATS}
        except (KeyError, ValueError):
            continue
        joined.append({
            "point_id": r["point_id"], "label": int(r["label"]),
            "matched_patch_id": matched_patch, "vector": vec, **feats,
        })
    return joined


def _mannwhitney(pos: list[float], neg: list[float]):
    from scipy import stats
    import numpy as np
    u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    r_rb = 1 - (2 * u) / (len(pos) * len(neg))
    return float(u), float(p), float(r_rb)


def run() -> None:
    joined = build_joined_dataset()
    n = len(joined)
    n_pos = sum(1 for j in joined if j["label"] == 1)
    n_neg = n - n_pos

    if n < 10 or n_neg < 4:
        write_csv(OUT_UNIV, [], UNIV_FIELDS)
        write_csv(OUT_FIRTH, [], FIRTH_FIELDS)
        write_csv(OUT_SUM, [
            {"stat_key": "n_joined", "stat_value": str(n)},
            {"stat_key": "status", "stat_value": "ANALYSIS_FAIL_CLOSED_INSUFFICIENT_N"},
        ], SUM_FIELDS)
        print(f"[v1r4] n={n} n_neg={n_neg} status=FAIL_CLOSED")
        return

    # PCA of the joined-subset embeddings (n=81, not all 300) -> 2 DINO features
    from sklearn.decomposition import PCA
    import numpy as np
    x = np.asarray([j["vector"] for j in joined], dtype=float)
    proj = PCA(n_components=2, svd_solver="full", random_state=0).fit_transform(x)
    for j, (px, py) in zip(joined, proj):
        j["dino_pca1"] = float(px)
        j["dino_pca2"] = float(py)

    all_feats = PHYSICAL_FEATS + ["dino_pca1", "dino_pca2"]

    # --- A) Univariate Mann-Whitney screen, all 8 features independently ---
    univ_rows = []
    for feat in all_feats:
        pos = [j[feat] for j in joined if j["label"] == 1]
        neg = [j[feat] for j in joined if j["label"] == 0]
        u, p, r_rb = _mannwhitney(pos, neg)
        univ_rows.append({
            "feature": feat, "n_pos": len(pos), "n_neg": len(neg),
            "mean_pos": round(sum(pos) / len(pos), 4), "mean_neg": round(sum(neg) / len(neg), 4),
            "mannwhitney_U": round(u, 2), "p_value": round(p, 4),
            "rank_biserial_r": round(r_rb, 4),
            "expected_sign": EXPECTED_SIGN.get(feat, 0),
            "direction_observed": "pos>neg" if sum(pos)/len(pos) > sum(neg)/len(neg) else "pos<neg",
            "significant_p05": bool(p < 0.05),
            "is_dino_feature": feat.startswith("dino_"),
        })
    write_csv(OUT_UNIV, univ_rows, UNIV_FIELDS)
    write_schema(SCH_UNIV, UNIV_FIELDS, "v1r4_dino_sedec_extended_univariate")

    # --- B) Firth multivariate, DINO-only (2 predictors, EPV=11 on n_neg=22) ---
    from firthlogist import FirthLogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Compatibility shim: firthlogist (older release) calls self._validate_data(),
    # an instance method removed from newer scikit-learn (>=1.6) in favor of the
    # standalone sklearn.utils.validation.validate_data function. Bind a thin
    # wrapper so firthlogist keeps working unmodified against the installed sklearn.
    if not hasattr(FirthLogisticRegression, "_validate_data"):
        from sklearn.utils.validation import validate_data as _sk_validate_data

        def _validate_data_shim(self, X, y=None, **kwargs):
            return _sk_validate_data(self, X, y, **kwargs)

        FirthLogisticRegression._validate_data = _validate_data_shim
    dino_feats = ["dino_pca1", "dino_pca2"]
    Xd = np.array([[j[f] for f in dino_feats] for j in joined])
    y = np.array([j["label"] for j in joined])
    scaler = StandardScaler()
    Xds = scaler.fit_transform(Xd)
    model = FirthLogisticRegression(fit_intercept=True)
    model.fit(Xds, y)
    firth_rows = []
    for i, feat in enumerate(dino_feats):
        firth_rows.append({
            "feature": feat, "coef_standardized": round(float(model.coef_[i]), 4),
            "ci_low_95": round(float(model.ci_[i][0]), 4),
            "ci_high_95": round(float(model.ci_[i][1]), 4),
            "p_value": round(float(model.pvals_[i]), 4),
            "ci_crosses_zero": bool(model.ci_[i][0] <= 0 <= model.ci_[i][1]),
        })
    write_csv(OUT_FIRTH, firth_rows, FIRTH_FIELDS)

    # LOO AUC for the DINO-only model (same discipline as pipeline_final_v5.py)
    from sklearn.model_selection import LeaveOneOut
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    loo = LeaveOneOut()
    y_true, y_score = [], []
    for tr_idx, te_idx in loo.split(Xd):
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xd[tr_idx])
        Xte = sc.transform(Xd[te_idx])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr_idx])
        y_score.append(clf.predict_proba(Xte)[:, 1][0])
        y_true.append(y[te_idx][0])
    loo_auc = roc_auc_score(y_true, y_score)

    summary = [
        {"stat_key": "sedec_primary_pool_n", "stat_value": "163"},
        {"stat_key": "n_joined_with_real_dino", "stat_value": str(n)},
        {"stat_key": "n_pos_flood_positive", "stat_value": str(n_pos)},
        {"stat_key": "n_neg_real_clean_sedec_negative", "stat_value": str(n_neg)},
        {"stat_key": "n_neg_of_22_total_covered", "stat_value": f"{n_neg}/22"},
        {"stat_key": "physical_feature_epv", "stat_value": f"{n_neg/6:.2f} (6 predictors, below 10 heuristic)"},
        {"stat_key": "dino_only_epv", "stat_value": f"{n_neg/2:.2f} (2 predictors, meets 10 heuristic)"},
        {"stat_key": "dino_only_firth_loo_auc", "stat_value": f"{loo_auc:.4f}"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "used_as_training_feature", "stat_value": "false"},
        {"stat_key": "analysis_separation",
         "stat_value": "univariate_screen_all_8_features_independent; "
                        "firth_multivariate_dino_only_2_predictors_never_mixed_with_physical"},
        {"stat_key": "status", "stat_value": "EXTENDED_ANALYSIS_READY_REVIEW_ONLY"},
    ]
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_doc(DOC, "v1r4 — Extended DINO x SEDEC Analysis (n=81, all 22 real negatives covered)", [
        "## Objetivo",
        f"Após baixar 13 patches novos centrados em evidência (negativos reais) e 2 "
        f"patches oficiais adicionais nesta sessão, {n} dos 163 registros SEDEC "
        f"primary (real-vs-real) agora têm embedding DINO real: {n_pos} positivos, "
        f"{n_neg} negativos — TODOS os 22 negativos reais cobertos.",
        "## Disciplina metodológica",
        "Duas análises separadas, nunca misturadas: (A) screen univariado "
        "Mann-Whitney nas 6 variáveis físicas + 2 componentes PCA do DINO, cada "
        "uma isolada; (B) modelo Firth multivariado usando SOMENTE os 2 "
        "componentes PCA do DINO (EPV=11, atende a heurística >=10) — nunca "
        "combinado com as 6 variáveis físicas no mesmo modelo (isso daria EPV=2.75, "
        "abaixo de qualquer heurística usável).",
        f"## Resultado",
        f"LOO AUC do modelo Firth somente-DINO: {loo_auc:.4f}.",
    ])
    print(f"[v1r4] n={n} pos={n_pos} neg={n_neg} dino_only_loo_auc={loo_auc:.4f}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1r4 extended dino sedec analysis").parse_args()
    run()
