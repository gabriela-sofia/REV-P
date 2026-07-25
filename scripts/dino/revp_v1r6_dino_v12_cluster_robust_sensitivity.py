"""REV-P v1r6 -- Sensibilidade a pseudorreplicacao do teste A/B v1r5.

v1r5 encontrou LRT significante (p=0.0048) para DINO no Modelo B, mas
descobriu que dino_pca1/2 sao por-PATCH, nao por-ponto: 109 pontos caem em
so 23 patches unicos (ate 10 pontos compartilhando o mesmo vetor). Este
script resolve essa lacuna com dois testes que respeitam explicitamente o
agrupamento por patch, em vez de assumir 109 observacoes independentes:

  A) Erro-padrao cluster-robusto (sandwich, clusterizado por patch_id) para
     os coeficientes do Modelo B (fisico + DINO) -- reestima a incerteza dos
     coeficientes DINO tratando os pontos do mesmo patch como nao-
     independentes, com correcao de pequena amostra (G clusters).
     Teste de Wald conjunto cluster-robusto para os 2 componentes DINO
     (analogo ao LRT de v1r5, mas robusto ao agrupamento).
  B) Correlacao descritiva por patch (n=23, um valor por patch): Spearman
     entre dino_pca1/dino_pca2 (fixo por patch) e a fracao de pontos
     positivos naquele patch -- nao e um teste formal (n pequeno), e
     reportado so como evidencia adicional.

Nao cria label. Nao treina nada novo. Reusa a mesma matriz joined de v1r5
(mesmo n=109, mesmos 23 patches, mesma escolha de features pelo orcamento de
EPV) para que a comparacao entre v1r5 e v1r6 seja direta.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

from revp_v1qg_v1qm_smoke_embedding_common import DATASETS, DOCS, _p, write_csv, write_doc
from revp_v1r5_dino_v12_ab_test import (
    ENV_IN_V12, ENV_SENTINEL_DIR, build_joined_dataset, choose_feature_budget, fit_firth,
)
from sklearn.decomposition import PCA

OUT_SUM = _p("REVP_V1R6_OUT_SUM", DATASETS / "dino_v12_cluster_robust_sensitivity_v1r6.csv")
DOC = _p("REVP_V1R6_DOC", DOCS / "revp_v1r6_dino_v12_cluster_robust_sensitivity.md")
SUM_FIELDS = ["stat_key", "stat_value"]


def cluster_robust_covariance(X: np.ndarray, y: np.ndarray, beta: np.ndarray,
                               cluster_ids: list[str]) -> np.ndarray:
    """Sandwich (CR0) covariance clustered by cluster_ids, with the standard
    small-sample correction c = (G/(G-1)) * ((n-1)/(n-p)). Uses the Firth fit
    (X'WX)^{-1} as the bread and per-cluster summed score residuals
    (y_i - pi_i) as the meat -- the standard clustered-sandwich construction
    for a fitted logistic-type model (Firth's small penalty term in the
    score is not re-included here; it is negligible next to the clustering
    correction itself, which is the effect being tested)."""
    eta = X @ beta
    pi = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(pi * (1 - pi), 1e-12, None)
    Xw = X * np.sqrt(w)[:, None]
    bread = np.linalg.inv(Xw.T @ Xw)

    u = y - pi
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(cluster_ids):
        groups.setdefault(g, []).append(i)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for idx in groups.values():
        xg = X[idx]
        ug = u[idx]
        score_g = xg.T @ ug
        meat += np.outer(score_g, score_g)

    n, p = X.shape
    G = len(groups)
    c = (G / (G - 1)) * ((n - 1) / (n - p))
    return c * (bread @ meat @ bread)


def run() -> None:
    in_v12_raw = os.environ.get(ENV_IN_V12, "")
    sentinel_dir_raw = os.environ.get(ENV_SENTINEL_DIR, "")
    if not in_v12_raw or not sentinel_dir_raw:
        write_csv(OUT_SUM, [{"stat_key": "status", "stat_value": "FAIL_CLOSED_MISSING_PRIVATE_PATH_ENV"}], SUM_FIELDS)
        print(f"[v1r6] FAIL_CLOSED: set {ENV_IN_V12} and {ENV_SENTINEL_DIR}.")
        return

    joined = build_joined_dataset(Path(in_v12_raw), Path(sentinel_dir_raw))
    n = len(joined)
    y = np.array([j["label"] for j in joined])
    n_pos = int(y.sum())
    n_neg = n - n_pos
    minority = min(n_pos, n_neg)

    x_emb = np.asarray([j["vector"] for j in joined], dtype=float)
    proj = PCA(n_components=2, svd_solver="full", random_state=0).fit_transform(x_emb)
    for j, (px, py) in zip(joined, proj):
        j["dino_pca1"] = float(px)
        j["dino_pca2"] = float(py)

    phys_feats, dino_k = choose_feature_budget(minority)
    dino_feats = ["dino_pca1", "dino_pca2"][:dino_k]
    model_b_feats = phys_feats + dino_feats

    Xr = np.array([[j[f] for f in model_b_feats] for j in joined])
    Xs = StandardScaler().fit_transform(Xr)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    res = fit_firth(Xd, y)
    cluster_ids = [j["matched_patch_id"] for j in joined]
    n_clusters = len(set(cluster_ids))

    V_cluster = cluster_robust_covariance(Xd, y, res["beta"], cluster_ids)
    se_cluster = np.sqrt(np.clip(np.diag(V_cluster), 0, None))
    z_cluster = res["beta"] / se_cluster
    p_cluster = 2 * (1 - stats.norm.cdf(np.abs(z_cluster)))

    feat_names = ["intercept"] + model_b_feats
    coef_rows = []
    for i, feat in enumerate(feat_names):
        coef_rows.append({
            "stat_key": f"coef__{feat}",
            "stat_value": f"beta={res['beta'][i]:.4f} se_naive={res['se'][i]:.4f} "
                           f"se_cluster={se_cluster[i]:.4f} p_naive={res['pvals'][i]:.4f} "
                           f"p_cluster={p_cluster[i]:.4f}",
        })

    # Cluster-robust joint Wald test for the DINO block (analogous to v1r5's LRT).
    dino_idx = [feat_names.index(f) for f in dino_feats]
    beta_dino = res["beta"][dino_idx]
    V_dino = V_cluster[np.ix_(dino_idx, dino_idx)]
    wald_stat = float(beta_dino @ np.linalg.solve(V_dino, beta_dino))
    wald_df = len(dino_idx)
    wald_p = float(1.0 - stats.chi2.cdf(wald_stat, df=wald_df))

    # Descriptive patch-level correlation (n=n_clusters, one row per patch).
    patch_vals: dict[str, dict[str, Any]] = {}
    for j in joined:
        pid = j["matched_patch_id"]
        d = patch_vals.setdefault(pid, {"labels": [], "dino_pca1": j["dino_pca1"], "dino_pca2": j["dino_pca2"]})
        d["labels"].append(j["label"])
    patch_frac_pos = np.array([np.mean(d["labels"]) for d in patch_vals.values()])
    patch_pca1 = np.array([d["dino_pca1"] for d in patch_vals.values()])
    patch_pca2 = np.array([d["dino_pca2"] for d in patch_vals.values()])
    rho1, p1 = stats.spearmanr(patch_pca1, patch_frac_pos)
    rho2, p2 = stats.spearmanr(patch_pca2, patch_frac_pos)

    verdict = ("DINO_SURVIVES_CLUSTER_ROBUST_CHECK" if wald_p < 0.05
               else "DINO_DOES_NOT_SURVIVE_CLUSTER_ROBUST_CHECK_CONSISTENT_WITH_PSEUDOREPLICATION_CONFOUND")

    summary = [
        {"stat_key": "n_points", "stat_value": str(n)},
        {"stat_key": "n_clusters_patches", "stat_value": str(n_clusters)},
        {"stat_key": "model_b_features", "stat_value": ",".join(model_b_feats)},
        {"stat_key": "small_sample_correction_c", "stat_value": f"{(n_clusters/(n_clusters-1))*((n-1)/(n-len(feat_names))):.4f}"},
    ] + coef_rows + [
        {"stat_key": "dino_joint_wald_cluster_robust_statistic", "stat_value": f"{wald_stat:.4f}"},
        {"stat_key": "dino_joint_wald_cluster_robust_df", "stat_value": str(wald_df)},
        {"stat_key": "dino_joint_wald_cluster_robust_p_value", "stat_value": f"{wald_p:.4f}"},
        {"stat_key": "v1r5_naive_lrt_p_value_for_comparison", "stat_value": "0.0048"},
        {"stat_key": "patch_level_spearman_dino_pca1_vs_frac_positive", "stat_value": f"rho={rho1:.4f} p={p1:.4f} n={n_clusters}"},
        {"stat_key": "patch_level_spearman_dino_pca2_vs_frac_positive", "stat_value": f"rho={rho2:.4f} p={p2:.4f} n={n_clusters}"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "used_as_training_feature", "stat_value": "false"},
        {"stat_key": "verdict", "stat_value": verdict},
        {"stat_key": "status", "stat_value": "CLUSTER_ROBUST_SENSITIVITY_READY_REVIEW_ONLY"},
    ]
    write_csv(OUT_SUM, summary, SUM_FIELDS)

    write_doc(DOC, "v1r6 -- Sensibilidade cluster-robusta ao achado DINO do v1r5", [
        "## Objetivo",
        "v1r5 achou LRT significante (p=0.0048) para DINO no Modelo B, mas "
        "descobriu que dino_pca1/2 sao por-patch (23 unicos para 109 "
        "pontos, ate 10 pontos compartilhando o mesmo vetor). Este script "
        "reestima a incerteza dos coeficientes DINO com erro-padrao "
        "cluster-robusto (sandwich, clusterizado por patch_id, correcao de "
        "pequena amostra), que e o jeito estatisticamente correto de "
        "tratar observacoes nao-independentes -- em vez de assumir que os "
        "109 pontos sao 109 informacoes DINO independentes.",
        f"## Resultado -- teste de Wald conjunto cluster-robusto para DINO",
        f"estatistica={wald_stat:.4f}, df={wald_df}, p={wald_p:.4f} "
        f"(vs. LRT ingenuo de v1r5: p=0.0048). "
        f"**Veredito: {verdict}.**",
        "## Correlacao descritiva por patch (n=23, so evidencia complementar)",
        f"Spearman dino_pca1 x fracao_positiva_no_patch: rho={rho1:.4f}, p={p1:.4f}. "
        f"Spearman dino_pca2 x fracao_positiva_no_patch: rho={rho2:.4f}, p={p2:.4f}. "
        f"n=23 e pequeno demais para um teste formal robusto -- reportado "
        f"so como leitura complementar, nao como prova.",
        "## Interpretacao",
        ("O sinal de v1r5 NAO sobrevive ao controle por clustering de "
         "patch: uma vez que a nao-independencia entre pontos do mesmo "
         "patch e contabilizada corretamente, a evidencia de que DINO "
         "agrega valor incremental ao modelo fisico deixa de ser "
         "estatisticamente solida com o n atual. Isso confirma a suspeita "
         "de v1r5 -- o resultado ingenuo era, ao menos em parte, um "
         "artefato de pseudorreplicacao, nao evidencia real de conteudo "
         "visual preditivo do DINO. **Conclusao honesta para o produto**: "
         "com os dados disponiveis agora (23 patches unicos com embedding "
         "real em Recife), o teste A/B nao suporta promover DINO a feature "
         "do score -- ele continua evidencia visual auxiliar/explicavel na "
         "interface, nunca input do modelo, ate que mais patches "
         "independentes com embedding real estejam disponiveis para "
         "refazer este teste com poder estatistico adequado."
         if wald_p >= 0.05 else
         "Mesmo apos o controle rigoroso por clustering de patch, o sinal "
         "DINO permanece estatisticamente robusto -- evidencia mais forte "
         "de que ha conteudo visual incremental real, nao apenas um "
         "artefato de patches repetidos. Ainda assim, o n de patches "
         "unicos (23) e pequeno; generalizar essa conclusao para fora da "
         "amostra atual exige mais patches independentes."),
        "## Limitacoes explicitas",
        f"n_clusters={n_clusters} e pequeno para inferencia cluster-robusta "
        f"assintotica (regra pratica usual pede G>=20-50 clusters; estamos "
        f"na borda inferior) -- a correcao de pequena amostra aplicada "
        f"(c={(n_clusters/(n_clusters-1))*((n-1)/(n-len(feat_names))):.3f}) "
        f"e um paliativo padrao, nao uma garantia. Nenhum label foi criado. "
        f"DINO continua evidencia auxiliar independentemente do veredito "
        f"deste script.",
    ])
    print(f"[v1r6] n={n} n_clusters={n_clusters} wald_stat={wald_stat:.4f} "
          f"wald_p={wald_p:.4f} verdict={verdict}")


if __name__ == "__main__":
    run()
