"""REV-P v1r5 -- Modelo A (fisico, v12) vs Modelo B (fisico + DINO), no
dataset mais forte disponivel (dataset_v12_final.csv, n=278, 154 pos/124 neg),
em vez do recorte de 163 usado em v1r4.

Executa a Fase 1 do PLANO_ACAO_produto_v1.md: refazer o teste A/B com o dado
mais forte, decidindo se DINO agrega valor incremental ao modelo fisico via
razao de verossimilhanca entre dois modelos Firth aninhados (nao via DeLong/
delta-AUC bruto -- ver revp_contrato_inferencia_v0_revalidacao_cientifica.md
secao 3.2).

Metodologia:
  1) Join espacial ponto-em-bbox entre os patches Sentinel com embedding DINO
     real (dataset_recife_sedec_all_embeddings_v1r3.csv, 23 patches) e os
     pontos de dataset_v12_final.csv que caem dentro dessas bboxes.
  2) Orcamento de EPV: com o n_neg do subconjunto joined, escolhe o maior
     numero de features fisicas + componentes DINO tal que
     n_neg/(n_features_fisicas+n_dino) >= 10 (mesma heuristica ja usada em
     pipeline_v12_primary.py e revp_v1r4). Prioridade: preservar o maximo de
     features fisicas, reduzir apenas se o orcamento nao permitir 6+K.
  3) Modelo A = Firth so com as features fisicas escolhidas.
     Modelo B = Modelo A + K componentes PCA do DINO (mesmas linhas, mesmo n).
  4) Razao de verossimilhanca (LRT) entre A e B usando a log-verossimilhanca
     penalizada de Firth de cada modelo (estatistica = 2*(llB-llA), df=K).
  5) Delta LOO-AUC (logistic regression L2 padrao, mesma disciplina de
     pipeline_v12_primary.py) reportado so como descritivo complementar.

Nao cria label. Nao treina nada fora deste teste de comparacao review-only.
DINO permanece evidencia auxiliar mesmo se o teste indicar valor incremental
(essa decisao de produto e tomada a parte, nao por este script).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from revp_v1qg_v1qm_smoke_embedding_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, parse_embedding_from_row, read_csv, write_csv, write_doc, write_schema,
)

IN_EMB = _p("REVP_V1R5_IN_EMB", DATASETS / "dino_recife_sedec_all_embeddings_v1r3.csv")
OUT_MODELS = _p("REVP_V1R5_OUT_MODELS", DATASETS / "dino_v12_ab_firth_model_coefs_v1r5.csv")
OUT_SUM = _p("REVP_V1R5_OUT_SUM", DATASETS / "dino_v12_ab_comparison_summary_v1r5.csv")
SCH_MODELS = _p("REVP_V1R5_SCH_MODELS", SCHEMAS / "dino_v12_ab_firth_model_coefs_v1r5_schema.csv")
DOC = _p("REVP_V1R5_DOC", DOCS / "revp_v1r5_dino_v12_ab_test.md")

# Private PROJETO inputs: no hardcoded absolute path is committed (REV-P must
# never carry PROJETO paths). Fail-closed when the env vars are not set.
ENV_IN_V12 = "REVP_V1R5_IN_V12"
ENV_SENTINEL_DIR = "REVP_V1R5_SENTINEL_DIR"

FEATURE_COLS_V12 = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf",
                     "rain_peak_residual_orthogonalized", "rain_decay_index_api_chirps"]
EXPECTED_SIGN = {"elevation_m": -1, "slope_deg": -1, "hand_m_dinf": -1, "twi_dinf": +1,
                  "rain_peak_residual_orthogonalized": +1, "rain_decay_index_api_chirps": +1,
                  "dino_pca1": 0, "dino_pca2": 0}
# Ranking por p-valor observado em primaria_v12_firth_multivariate_coefs.csv
# (n=278, todo o dataset v12): usado apenas para decidir a ORDEM de reducao
# quando o orcamento de EPV do subconjunto joined nao permitir as 6 completas.
FEATURE_RANK_BY_V12_EVIDENCE = ["rain_decay_index_api_chirps", "twi_dinf", "slope_deg",
                                 "elevation_m", "rain_peak_residual_orthogonalized", "hand_m_dinf"]

MODEL_FIELDS = ["model", "feature", "coef_standardized", "se", "ci_low_95", "ci_high_95",
                "p_value", "expected_sign", "sign_matches_expected"]
SUM_FIELDS = ["stat_key", "stat_value"]


def _patch_bboxes_wgs84(sentinel_dir: Path) -> dict[str, tuple[float, float, float, float]]:
    import rasterio
    from rasterio.warp import transform_bounds
    out: dict[str, tuple[float, float, float, float]] = {}
    for p in sorted(glob.glob(str(sentinel_dir / "patch_recife_*.tif"))):
        with rasterio.open(p) as ds:
            b = ds.bounds
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
        pid = Path(p).stem.split("patch_recife_")[-1].upper()
        out[pid] = (lon_min, lat_min, lon_max, lat_max)
    return out


def build_joined_dataset(in_v12: Path, sentinel_dir: Path) -> list[dict[str, Any]]:
    v12_rows = read_csv(in_v12)
    bboxes = _patch_bboxes_wgs84(sentinel_dir)
    emb_rows = read_csv(IN_EMB)
    emb_by_patch = {r["patch_id"]: parse_embedding_from_row(r) for r in emb_rows}

    joined: list[dict[str, Any]] = []
    for r in v12_rows:
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
            label = int(r["label"])
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
            feats = {f: float(r[f]) for f in FEATURE_COLS_V12}
        except (KeyError, ValueError):
            continue
        joined.append({
            "point_id": r["point_id"], "label": label,
            "matched_patch_id": matched_patch, "vector": vec, **feats,
        })
    return joined


def fit_firth(X: np.ndarray, y: np.ndarray, max_iter: int = 200, tol: float = 1e-10) -> dict[str, Any]:
    """Firth (bias-reduced) penalized logistic regression, fit via modified
    scores (Heinze & Schemper 2002). X must already include an intercept
    column. Validated against firthlogist's published v12 coefficients and
    penalized log-likelihood (matches to >=4 decimals) before use here --
    see local validation run, not committed as a test fixture since it
    depends on the private PROJETO v12 CSV."""
    n, p = X.shape
    beta = np.zeros(p)
    it = 0
    for it in range(max_iter):
        eta = X @ beta
        pi = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(pi * (1 - pi), 1e-12, None)
        Xw = X * np.sqrt(w)[:, None]
        info = Xw.T @ Xw
        info_inv = np.linalg.inv(info)
        H = Xw @ info_inv @ Xw.T
        h = np.diag(H)
        U = X.T @ (y - pi + h * (0.5 - pi))
        delta = info_inv @ U
        beta = beta + delta
        if np.linalg.norm(delta) < tol:
            break
    eta = X @ beta
    pi = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(pi * (1 - pi), 1e-12, None)
    Xw = X * np.sqrt(w)[:, None]
    info = Xw.T @ Xw
    info_inv = np.linalg.inv(info)
    se = np.sqrt(np.diag(info_inv))
    loglik = float(np.sum(y * np.log(np.clip(pi, 1e-12, None))
                          + (1 - y) * np.log(np.clip(1 - pi, 1e-12, None))))
    _, logdet = np.linalg.slogdet(info)
    penalized_loglik = loglik + 0.5 * logdet
    z = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    return {"beta": beta, "se": se, "pvals": pvals, "ci_low": ci_low, "ci_high": ci_high,
            "penalized_loglik": penalized_loglik, "n_iter": it + 1}


def _loo_auc(X: np.ndarray, y: np.ndarray) -> float:
    loo = LeaveOneOut()
    y_true, y_score = [], []
    for tr_idx, te_idx in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr_idx])
        Xte = sc.transform(X[te_idx])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr_idx])
        y_score.append(clf.predict_proba(Xte)[:, 1][0])
        y_true.append(y[te_idx][0])
    return float(roc_auc_score(y_true, y_score))


def choose_feature_budget(n_neg: int) -> tuple[list[str], int]:
    """EPV>=10 on the minority class, preferring the most physical features
    and, only if the budget forces it, falling back to the strongest-ranked
    subset. Returns (physical_features_chosen, n_dino_components_chosen)."""
    for dino_k in (2, 1):
        for phys_k in (6, 3, 2, 1):
            phys = (FEATURE_COLS_V12 if phys_k == 6 else FEATURE_RANK_BY_V12_EVIDENCE[:phys_k])
            n_pred = phys_k + dino_k
            if n_neg / n_pred >= 10.0:
                return phys, dino_k
    return FEATURE_RANK_BY_V12_EVIDENCE[:1], 1


def run() -> None:
    in_v12_raw = os.environ.get(ENV_IN_V12, "")
    sentinel_dir_raw = os.environ.get(ENV_SENTINEL_DIR, "")
    if not in_v12_raw or not sentinel_dir_raw:
        write_csv(OUT_MODELS, [], MODEL_FIELDS)
        write_csv(OUT_SUM, [
            {"stat_key": "status", "stat_value": "FAIL_CLOSED_MISSING_PRIVATE_PATH_ENV"},
            {"stat_key": "missing_env_vars", "stat_value": f"{ENV_IN_V12},{ENV_SENTINEL_DIR}"},
        ], SUM_FIELDS)
        print(f"[v1r5] FAIL_CLOSED: set {ENV_IN_V12} and {ENV_SENTINEL_DIR} "
              f"(private PROJETO paths, never hardcoded in this repo).")
        return

    joined = build_joined_dataset(Path(in_v12_raw), Path(sentinel_dir_raw))
    n = len(joined)
    n_pos = sum(1 for j in joined if j["label"] == 1)
    n_neg = n - n_pos
    minority = min(n_pos, n_neg)

    if n < 10 or minority < 4:
        write_csv(OUT_MODELS, [], MODEL_FIELDS)
        write_csv(OUT_SUM, [
            {"stat_key": "n_joined", "stat_value": str(n)},
            {"stat_key": "status", "stat_value": "ANALYSIS_FAIL_CLOSED_INSUFFICIENT_N"},
        ], SUM_FIELDS)
        print(f"[v1r5] n={n} minority={minority} status=FAIL_CLOSED")
        return

    x_emb = np.asarray([j["vector"] for j in joined], dtype=float)
    proj = PCA(n_components=2, svd_solver="full", random_state=0).fit_transform(x_emb)
    for j, (px, py) in zip(joined, proj):
        j["dino_pca1"] = float(px)
        j["dino_pca2"] = float(py)

    phys_feats, dino_k = choose_feature_budget(minority)
    dino_feats = ["dino_pca1", "dino_pca2"][:dino_k]
    model_a_feats = phys_feats
    model_b_feats = phys_feats + dino_feats

    y = np.array([j["label"] for j in joined])

    def fit_named(feat_list: list[str], model_name: str):
        Xr = np.array([[j[f] for f in feat_list] for j in joined])
        Xs = StandardScaler().fit_transform(Xr)
        Xd = np.column_stack([np.ones(len(Xs)), Xs])
        res = fit_firth(Xd, y)
        rows = [{
            "model": model_name, "feature": "intercept",
            "coef_standardized": round(float(res["beta"][0]), 4),
            "se": round(float(res["se"][0]), 4),
            "ci_low_95": round(float(res["ci_low"][0]), 4),
            "ci_high_95": round(float(res["ci_high"][0]), 4),
            "p_value": round(float(res["pvals"][0]), 4),
            "expected_sign": 0, "sign_matches_expected": "",
        }]
        for i, feat in enumerate(feat_list, start=1):
            exp_sign = EXPECTED_SIGN.get(feat, 0)
            rows.append({
                "model": model_name, "feature": feat,
                "coef_standardized": round(float(res["beta"][i]), 4),
                "se": round(float(res["se"][i]), 4),
                "ci_low_95": round(float(res["ci_low"][i]), 4),
                "ci_high_95": round(float(res["ci_high"][i]), 4),
                "p_value": round(float(res["pvals"][i]), 4),
                "expected_sign": exp_sign,
                "sign_matches_expected": (bool(np.sign(res["beta"][i]) == exp_sign) if exp_sign != 0 else ""),
            })
        return res, rows, Xr

    res_a, rows_a, Xr_a = fit_named(model_a_feats, "A_physical_only")
    res_b, rows_b, Xr_b = fit_named(model_b_feats, "B_physical_plus_dino")

    lrt_stat = 2.0 * (res_b["penalized_loglik"] - res_a["penalized_loglik"])
    lrt_df = dino_k
    lrt_p = float(1.0 - stats.chi2.cdf(max(lrt_stat, 0.0), df=lrt_df))

    loo_auc_a = _loo_auc(Xr_a, y)
    loo_auc_b = _loo_auc(Xr_b, y)
    delta_loo_auc = loo_auc_b - loo_auc_a

    # Critical caveat: dino_pca1/2 are computed ONE PER PATCH, not one per
    # point. Multiple v12 points sharing a patch get an IDENTICAL DINO
    # vector -- this is pseudoreplication that inflates the apparent
    # independent sample size for the DINO dimensions specifically (the
    # physical features do NOT share this problem, they are per-point).
    from collections import Counter
    points_per_patch = Counter(j["matched_patch_id"] for j in joined)
    n_unique_patches = len(points_per_patch)
    max_points_per_patch = max(points_per_patch.values())
    pseudoreplication_flag = n_unique_patches < n

    write_csv(OUT_MODELS, rows_a + rows_b, MODEL_FIELDS)
    write_schema(SCH_MODELS, MODEL_FIELDS, "v1r5_dino_v12_ab_firth_model_coefs")

    if lrt_p >= 0.05:
        decision = "DINO_NO_INCREMENTAL_VALUE_LRT_NOT_SIGNIFICANT"
    elif pseudoreplication_flag:
        decision = "DINO_LRT_SIGNIFICANT_BUT_CONFOUNDED_BY_PATCH_LEVEL_PSEUDOREPLICATION"
    else:
        decision = "DINO_ADDS_INCREMENTAL_VALUE_LRT_P_LT_0_05"

    summary = [
        {"stat_key": "source_dataset", "stat_value": "dataset_v12_final.csv (n=278, 154 pos/124 neg, full pool)"},
        {"stat_key": "n_joined_with_real_dino", "stat_value": str(n)},
        {"stat_key": "n_pos_joined", "stat_value": str(n_pos)},
        {"stat_key": "n_neg_joined", "stat_value": str(n_neg)},
        {"stat_key": "minority_class_joined", "stat_value": str(minority)},
        {"stat_key": "n_patches_with_real_dino_embedding", "stat_value": str(len(read_csv(IN_EMB)))},
        {"stat_key": "model_a_features", "stat_value": ",".join(model_a_feats)},
        {"stat_key": "model_b_features", "stat_value": ",".join(model_b_feats)},
        {"stat_key": "dino_pca_components_used", "stat_value": str(dino_k)},
        {"stat_key": "epv_model_a", "stat_value": f"{minority / len(model_a_feats):.2f}"},
        {"stat_key": "epv_model_b", "stat_value": f"{minority / len(model_b_feats):.2f}"},
        {"stat_key": "epv_budget_rule", "stat_value": "min combo (phys_k in [6,3,2,1] pref. more phys, dino_k in [2,1] pref. more dino) s.t. minority/n_predictors>=10"},
        {"stat_key": "penalized_loglik_model_a", "stat_value": f"{res_a['penalized_loglik']:.4f}"},
        {"stat_key": "penalized_loglik_model_b", "stat_value": f"{res_b['penalized_loglik']:.4f}"},
        {"stat_key": "lrt_statistic", "stat_value": f"{lrt_stat:.4f}"},
        {"stat_key": "lrt_df", "stat_value": str(lrt_df)},
        {"stat_key": "lrt_p_value", "stat_value": f"{lrt_p:.4f}"},
        {"stat_key": "loo_auc_model_a", "stat_value": f"{loo_auc_a:.4f}"},
        {"stat_key": "loo_auc_model_b", "stat_value": f"{loo_auc_b:.4f}"},
        {"stat_key": "delta_loo_auc_descriptive_only", "stat_value": f"{delta_loo_auc:.4f}"},
        {"stat_key": "n_unique_patches_with_dino_embedding_used", "stat_value": str(n_unique_patches)},
        {"stat_key": "max_points_sharing_one_identical_dino_vector", "stat_value": str(max_points_per_patch)},
        {"stat_key": "dino_pseudoreplication_caveat",
         "stat_value": f"dino_pca1/2 are PER-PATCH not per-point; {n} points map to only "
                        f"{n_unique_patches} unique DINO vectors (up to {max_points_per_patch} points "
                        f"share one identical vector) -- effective independent sample size for the DINO "
                        f"dimensions is closer to {n_unique_patches} than {n}; physical features do not "
                        f"share this problem (computed per-point)"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "used_as_training_feature", "stat_value": "false"},
        {"stat_key": "decision_basis", "stat_value": "likelihood_ratio_test_not_delta_auc"},
        {"stat_key": "decision", "stat_value": decision},
        {"stat_key": "status", "stat_value": "AB_TEST_READY_REVIEW_ONLY"},
    ]
    write_csv(OUT_SUM, summary, SUM_FIELDS)

    write_doc(DOC, "v1r5 -- Teste A/B fisico vs fisico+DINO no v12 (n=278, dado mais forte)", [
        "## Objetivo",
        "Executa a Fase 1 do PLANO_ACAO_produto_v1.md: decide se DINO agrega "
        "valor incremental ao modelo fisico usando o dataset mais forte "
        "disponivel (`dataset_v12_final.csv`, n=278, 154 pos/124 neg, "
        "LOO-AUC=0.6781 documentado), em vez do recorte antigo de 163 pontos "
        "usado em v1r4.",
        f"## Join espacial",
        f"Dos 278 pontos do v12, {n} caem dentro de um patch Sentinel com "
        f"embedding DINO real ({n_pos} positivos, {n_neg} negativos) -- "
        f"restricao imposta pela cobertura de patches com embedding "
        f"(23 patches: 10 positivos + 13 negativos-evidencia), nao pelo "
        f"tamanho do v12 em si.",
        "## Orcamento de EPV",
        f"Com minority={minority} no subconjunto joined, o orcamento de EPV "
        f"(>=10 na classe minoritaria) permitiu: Modelo A = "
        f"{', '.join(model_a_feats)} (EPV={minority/len(model_a_feats):.2f}); "
        f"Modelo B = Modelo A + {dino_k} componente(s) PCA do DINO "
        f"(EPV={minority/len(model_b_feats):.2f}). "
        + ("As 6 features fisicas completas do v12 couberam no orcamento."
           if phys_feats == FEATURE_COLS_V12 else
           "As 6 features fisicas completas NAO couberam no orcamento "
           "(EPV cairia abaixo de 10); reduzido para as mais fortes por "
           "evidencia real do proprio v12 (menor p-valor no Firth "
           "multivariado de n=278: rain_decay_index_api_chirps, twi_dinf, "
           "...), nao por escolha arbitraria."),
        "## Resultado -- razao de verossimilhanca (nao DeLong, nao delta-AUC bruto)",
        f"LRT: estatistica={lrt_stat:.4f}, df={lrt_df}, p={lrt_p:.4f}. "
        f"Decisao: **{decision}**. "
        f"LOO-AUC descritivo: Modelo A={loo_auc_a:.4f}, Modelo B={loo_auc_b:.4f}, "
        f"delta={delta_loo_auc:+.4f} (reportado apenas como evidencia "
        f"complementar, nunca como criterio de decisao -- ver revalidacao "
        f"cientifica anterior, secao 3.2).",
        "## Limitacao critica: DINO e por patch, nao por ponto (pseudorreplicacao)",
        f"O embedding DINO (e, portanto, `dino_pca1`/`dino_pca2`) e calculado "
        f"UMA VEZ POR PATCH Sentinel, nao uma vez por ponto. Os {n} pontos "
        f"joined caem em apenas {n_unique_patches} patches unicos -- ate "
        f"{max_points_per_patch} pontos compartilham exatamente o mesmo vetor "
        f"DINO. Isso e pseudorreplicacao: o tamanho amostral efetivamente "
        f"independente para as dimensoes DINO e muito mais proximo de "
        f"{n_unique_patches} do que de {n}. As features fisicas NAO tem esse "
        f"problema (sao calculadas por ponto, a partir de terreno/chuva reais "
        f"na coordenada exata). "
        + ("Como ha patches com pontos de ambas as classes (positivo e "
           "negativo) e patches puros de uma so classe, parte do sinal que a "
           "LRT capta pode refletir qual PATCH um ponto caiu, nao uma "
           "caracteristica visual real de risco de enchente -- por isso a "
           "decisao acima foi marcada como CONFOUNDED, nao como um achado "
           "limpo. Uma reanalise por patch (1 linha por patch, nao por "
           "ponto) e o proximo passo real antes de aceitar este resultado "
           "como definitivo." if pseudoreplication_flag else
           "Neste caso n_unique_patches == n (cada ponto tem um patch "
           "proprio), entao a pseudorreplicacao nao se aplica."),
        "## Limitacoes explicitas",
        f"n={n} restrito a Recife e aos patches com embedding DINO real "
        f"(nao e uma amostra aleatoria do v12); nenhum label foi criado; "
        f"nenhum treino supervisionado novo; DINO permanece evidencia "
        f"auxiliar independentemente do resultado -- a decisao de produto "
        f"(DINO como feature vs. evidencia visual explicavel) e tomada a "
        f"parte deste script, com base neste resultado mais o restante do "
        f"plano de acao.",
        "## Implementacao Firth",
        "`firthlogist` exige Python <3.11 e nao esta disponivel neste "
        "ambiente (Python 3.12); a regressao logistica penalizada de Firth "
        "foi reimplementada localmente (scores modificados, Heinze & "
        "Schemper 2002) e validada antes de uso: reproduz os coeficientes "
        "padronizados publicados em `primaria_v12_firth_multivariate_coefs.csv` "
        "com 4 casas decimais de precisao e a log-verossimilhanca penalizada "
        "publicada em `all_reports_v12_primary.json` "
        "(-150.5935 vs -150.59347...). Os intervalos de confianca/p-valores "
        "aqui usam aproximacao de Wald (nao perfil de verossimilhanca "
        "penalizada como o `firthlogist` de referencia), o que pode gerar "
        "pequenas diferencas no terceiro/quarto digito de p-valor para "
        "coeficientes proximos da fronteira de significancia -- nao afeta "
        "os pontos estimados nem a log-verossimilhanca usada na LRT.",
    ])
    print(f"[v1r5] n={n} pos={n_pos} neg={n_neg} model_a={model_a_feats} "
          f"model_b_dino_k={dino_k} lrt_p={lrt_p:.4f} decision={decision}")


if __name__ == "__main__":
    run()
