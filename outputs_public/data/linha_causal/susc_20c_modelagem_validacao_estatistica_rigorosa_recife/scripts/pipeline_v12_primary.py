#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUSC-20 — pipeline causal de Recife, versao consolidada e certeira.

Um script, um dataset de entrada, um conjunto de resultados. Substitui a
cadeia de 4 scripts espalhados em 2 pastas (susc_20b/scripts/*.py +
susc_20c/scripts/pipeline_v12_primary.py), cujo script de modelagem apontava
pra um caminho local (`local_runs/...`) que nao existe no repositorio - por
isso, como estava, NAO DAVA PRA RODAR a partir de um clone limpo. Esta versao
le e escreve so em caminhos que existem de verdade no repo.

Metodologia (identica ao original, so o encanamento de arquivo mudou):
screening univariado (Mann-Whitney), regressao logistica penalizada de Firth
multivariada, bootstrap estratificado N=1000 (CIs e taxa de sign-flip), e AUC
preditivo (leave-one-out + 5-fold repetido 50x).

Verificado: rodando este script contra o dataset ja publicado no repo, os
numeros batem exatamente com os que ja estao documentados no README e no
relatorio v12 master (LOO-AUC=0.6781, n=269, mesmos coeficientes de Firth) -
ver `docs/reproducibilidade_susc20_recife.md`.

Ambiente necessario (ver environment.yml): Python 3.10, `scikit-learn<1.6`
(versoes mais novas quebram `firthlogist` - `_validate_data` foi removido da
API interna do sklearn) e `firthlogist`.

Roda a partir da raiz do repo:

    python outputs_public/data/linha_causal/susc_20c_modelagem_validacao_estatistica_rigorosa_recife/scripts/pipeline_v12_primary.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = next(
    p for p in (Path(__file__).resolve(), *Path(__file__).resolve().parents)
    if (p / ".git").is_dir() and (p / "environment.yml").is_file()
)
DATASET = ROOT / "outputs_public" / "data" / "linha_causal" / "susc_20a_aquisicao_eventos_reais_recife" / "dataset" / "dataset_eventos_features_v12_final.csv"
RESULTS_DIR = ROOT / "outputs_public" / "data" / "linha_causal" / "susc_20c_modelagem_validacao_estatistica_rigorosa_recife" / "results"

SEED = 20260723
FEATURE_COLS = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
                 "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
EXPECTED_SIGN = {"elevation_m": -1, "slope_deg": -1, "hand_m_dinf": -1, "twi_dinf": +1,
                  "rain_peak_residual_orthogonalized": +1, "rain_decay_index_api_chirps": +1}


def univariate_screen(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        d = df.dropna(subset=[feat])
        pos = d.loc[d["label"] == 1, feat].values
        neg = d.loc[d["label"] == 0, feat].values
        if len(pos) < 2 or len(neg) < 2:
            continue
        u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        r_rb = 1 - (2 * u) / (len(pos) * len(neg))
        rows.append({"feature": feat, "n_pos": len(pos), "n_neg": len(neg),
                      "mean_pos": round(float(np.mean(pos)), 4), "mean_neg": round(float(np.mean(neg)), 4),
                      "mannwhitney_U": round(float(u), 2), "p_value": round(float(p), 4),
                      "rank_biserial_r": round(float(r_rb), 4),
                      "expected_sign": EXPECTED_SIGN.get(feat),
                      "direction_observed": "pos>neg" if np.mean(pos) > np.mean(neg) else "pos<neg",
                      "significant_p05": bool(p < 0.05)})
    return pd.DataFrame(rows)


def firth_multivariate(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    from firthlogist import FirthLogisticRegression
    d = df.dropna(subset=feature_cols).copy()
    X = d[feature_cols].values
    y = d["label"].astype(int).values
    Xs = StandardScaler().fit_transform(X)
    model = FirthLogisticRegression(fit_intercept=True)
    model.fit(Xs, y)
    rows = []
    for i, feat in enumerate(feature_cols):
        rows.append({"feature": feat, "coef_standardized": round(float(model.coef_[i]), 4),
                      "ci_low_95": round(float(model.ci_[i][0]), 4),
                      "ci_high_95": round(float(model.ci_[i][1]), 4),
                      "p_value": round(float(model.pvals_[i]), 4),
                      "expected_sign": EXPECTED_SIGN.get(feat),
                      "sign_matches_expected": bool(np.sign(model.coef_[i]) == EXPECTED_SIGN.get(feat, 0)),
                      "ci_crosses_zero": bool(model.ci_[i][0] <= 0 <= model.ci_[i][1])})
    coef_df = pd.DataFrame(rows)
    report = {"n_used": int(len(d)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
              "events_per_predictor_minority_class": round((y == 0).sum() / len(feature_cols), 2),
              "loglik": float(model.loglik_)}
    return coef_df, report


def bootstrap_firth_coefs(df: pd.DataFrame, feature_cols: list[str], n_boot: int = 1000, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    from firthlogist import FirthLogisticRegression
    d = df.dropna(subset=feature_cols).reset_index(drop=True)
    X = d[feature_cols].values
    y = d["label"].astype(int).values
    pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    boot_coefs = {f: [] for f in feature_cols}
    n_failed = 0
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos_idx, size=len(pos_idx), replace=True),
                              rng.choice(neg_idx, size=len(neg_idx), replace=True)])
        Xb, yb = X[bi], y[bi]
        try:
            Xbs = StandardScaler().fit_transform(Xb)
            m = FirthLogisticRegression(fit_intercept=True, skip_ci=True, skip_pvals=True)
            m.fit(Xbs, yb)
            for i, f in enumerate(feature_cols):
                boot_coefs[f].append(float(m.coef_[i]))
        except Exception:
            n_failed += 1
    rows = []
    for f in feature_cols:
        arr = np.array(boot_coefs[f])
        point_sign = np.sign(arr.mean())
        flip_pct = 100.0 * float(np.mean(np.sign(arr) != point_sign))
        ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
        rows.append({"feature": f, "n_boot_success": len(arr), "boot_mean_coef": round(float(arr.mean()), 4),
                      "boot_ci_low_2.5pct": round(float(ci_lo), 4), "boot_ci_high_97.5pct": round(float(ci_hi), 4),
                      "ci_crosses_zero": bool(ci_lo <= 0 <= ci_hi), "pct_sign_flips": round(flip_pct, 1)})
    return pd.DataFrame(rows), {"n_boot_requested": n_boot, "n_boot_failed": n_failed, "seed": seed}


def predictive_auc(df: pd.DataFrame, feature_cols: list[str], k: int = 5, n_repeats: int = 50, seed: int = SEED) -> dict:
    d = df.dropna(subset=feature_cols).reset_index(drop=True)
    X, y = d[feature_cols].values, d["label"].astype(int).values

    def _fit_score(tr_idx, te_idx):
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(X[tr_idx]), scaler.transform(X[te_idx])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr_idx])
        return clf.predict_proba(Xte)[:, 1]

    y_true, y_score = [], []
    for tr_idx, te_idx in LeaveOneOut().split(X):
        y_score.append(_fit_score(tr_idx, te_idx)[0])
        y_true.append(y[te_idx][0])
    loo_auc = roc_auc_score(y_true, y_score)

    rng = np.random.default_rng(seed)
    reps_auc = []
    for _ in range(n_repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))
        yt, ys = [], []
        for tr_idx, te_idx in skf.split(X, y):
            ys.extend(_fit_score(tr_idx, te_idx))
            yt.extend(y[te_idx])
        reps_auc.append(roc_auc_score(yt, ys))
    reps_auc = np.array(reps_auc)

    return {"n_used": int(len(d)), "loo_auc": round(float(loo_auc), 4), "skf_k": k, "skf_n_repeats": n_repeats,
            "skf_auc_mean": round(float(reps_auc.mean()), 4), "skf_auc_std": round(float(reps_auc.std()), 4),
            "skf_auc_min": round(float(reps_auc.min()), 4), "skf_auc_max": round(float(reps_auc.max()), 4)}


def main():
    primary = pd.read_csv(DATASET)
    print(f"primary n={len(primary)} ({(primary.label == 1).sum()} pos / {(primary.label == 0).sum()} neg)")

    print("[1] screening univariado (Mann-Whitney)")
    univ_df = univariate_screen(primary, FEATURE_COLS)
    univ_df.to_csv(RESULTS_DIR / "primaria_v12_univariate_mannwhitney.csv", index=False)

    print("[2] Firth multivariada")
    firth_df, firth_report = firth_multivariate(primary, FEATURE_COLS)
    firth_df.to_csv(RESULTS_DIR / "primaria_v12_firth_multivariate_coefs.csv", index=False)
    print(json.dumps(firth_report, indent=2))

    print("[3] bootstrap (N=1000)")
    boot_df, boot_report = bootstrap_firth_coefs(primary, FEATURE_COLS)
    boot_df.to_csv(RESULTS_DIR / "primaria_v12_bootstrap_coefs.csv", index=False)

    print("[4] AUC preditivo (LOO + 5-fold repetido 50x)")
    auc_report = predictive_auc(primary, FEATURE_COLS)
    with open(RESULTS_DIR / "primaria_v12_predictive_auc.json", "w") as f:
        json.dump(auc_report, f, indent=2)
    print(json.dumps(auc_report, indent=2))

    with open(RESULTS_DIR / "all_reports_v12_primary.json", "w") as f:
        json.dump({"firth_report": firth_report, "boot_report": boot_report, "auc_report": auc_report}, f, indent=2, default=str)

    print("\nDONE.")


if __name__ == "__main__":
    main()
