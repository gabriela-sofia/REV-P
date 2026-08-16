"""
MOD-RECIFE-03 -- Reajuste do modelo pluvial de Recife apos fonte de chuva
unica (`chuva02_padronizar_fonte_unica_recife.py`).

DE ONDE VEIO E O QUE COMPARA:
o v12 (PROJETO/local_runs/recife_modelo_v12_extracao_final/pipeline_v12_primary.py)
e a rota primaria publicada de Recife: Firth penalizado, 6 features causais,
LOO-AUC = 0,6781 (n=278, 154 pos / 124 neg). `aud_chuva01` mediu que a coluna
de chuva daquele ajuste misturava CHIRPS (181 pontos) e Open-Meteo/ERA5-Land
(97 pontos) na mesma coluna, com AUC do indicador de fonte (0,826) maior que
o da propria chuva (0,738) -- parte do que o coeficiente de chuva medida nao
era chuva, era proveniencia.

Este script REPETE A MESMA metodologia do v12 primario -- mesmas 6 features,
mesmo Firth, mesmo LOO, mesma semente onde aplicavel -- trocando so a fonte de
chuva pela unica (Open-Meteo/ERA5-Land, decisao de 2026-08-16, ver
`chuva02_padronizar_fonte_unica_recife.py`). Isola uma variavel por vez: nao
troca a resolucao do terreno junto (fica em NATIVA, a mesma do v12 original) --
isso e outra decisao, ja tomada e documentada em
`ext_resolucao_unica_30m_v2.md`, e misturar as duas mudancas no mesmo
antes/depois tornaria o efeito de nenhuma das duas legivel.

NAO faz: nao decide se o v12 fica invalidado (isso e leitura do numero, nao
decisao automatica deste script); nao promove esta rota para o pool fluvial
(Recife continua PLUVIAL_URBANO, fora do pool, por mecanismo); nao toca
PROJETO (le so para comparar o numero publicado, nao para escrever).

Uso:
    python scripts/suscetibilidade/mod_recife03_pluvial_fonte_unica.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
HARM = REPO / "local_runs" / "ter-03-brasil-harmonizado" / "recife_harmonizado.csv"
OUT = REPO / "local_runs" / "mod-recife-03-pluvial-fonte-unica"
SEED = 20260723  # mesma semente do v12 primario, para nao introduzir variacao extra

FEATURES = ["elevation_m_original", "slope_deg_original", "hand_m_dinf_original",
            "twi_dinf_original", "rain_peak_residual_orthogonalized",
            "rain_decay_index_api_chirps"]
SINAL_ESPERADO = {"elevation_m_original": -1, "slope_deg_original": -1,
                  "hand_m_dinf_original": -1, "twi_dinf_original": +1,
                  "rain_peak_residual_orthogonalized": +1,
                  "rain_decay_index_api_chirps": +1}

# publicado em primaria_v12_firth_multivariate_coefs.csv / all_reports_v12_primary.json
# (PROJETO, lido para comparar -- nao alterado, nao reescrito)
V12_PUBLICADO = {
    "loo_auc": 0.6781, "n": 278, "pos": 154, "neg": 124,
    "coef": {
        "elevation_m": {"coef": 0.2662, "p": 0.3738, "ic": [-0.3208, 0.8718]},
        "slope_deg": {"coef": -0.1698, "p": 0.2236, "ic": [-0.457, 0.1024]},
        "hand_m_dinf": {"coef": -0.0001, "p": 0.9781, "ic": [-0.5972, 0.5863]},
        "twi_dinf": {"coef": 0.2786, "p": 0.0461, "ic": [0.0047, 0.5693]},
        "rain_peak_residual_orthogonalized": {"coef": -0.1402, "p": 0.3467,
                                              "ic": [-0.4217, 0.1628]},
        "rain_decay_index_api_chirps": {"coef": 0.9896, "p": 0.0001,
                                        "ic": [0.6133, 1.4231]},
    },
}


def orthogonalizar(rain_max: pd.Series, decay: pd.Series) -> tuple[pd.Series, dict]:
    """Mesma formula do v12 (`build_v12_dataset.py:orthogonalize_single`),
    agora GLOBAL -- com fonte unica nao ha mais 'por fonte' para agrupar."""
    x = decay.to_numpy(dtype="float64")
    y = rain_max.to_numpy(dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    xm = x[ok].mean()
    beta = float(((x[ok] - xm) @ y[ok]) / ((x[ok] - xm) @ (x[ok] - xm)))
    intercept = float(y[ok].mean() - beta * xm)
    resid = pd.Series(np.nan, index=rain_max.index)
    pred = intercept + beta * x[ok]
    resid.iloc[np.where(ok)[0]] = y[ok] - pred
    corr_antes = float(pd.Series(x[ok]).corr(pd.Series(y[ok])))
    corr_depois = float(pd.Series(x[ok]).corr(pd.Series(resid.to_numpy()[ok])))
    return resid, {"n_usado": int(ok.sum()), "beta": round(beta, 4),
                   "intercept": round(intercept, 4),
                   "pearson_antes": round(corr_antes, 4),
                   "pearson_depois": round(corr_depois, 6)}


def firth_multivariado(d: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    import susc_firth_shim  # noqa: F401  -- aplica compat antes do import abaixo
    from firthlogist import FirthLogisticRegression

    dd = d.dropna(subset=features + ["label"]).copy()
    X = dd[features].to_numpy(dtype="float64")
    y = dd["label"].astype(int).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    m = FirthLogisticRegression(fit_intercept=True)
    m.fit(Xs, y)
    linhas = []
    for i, f in enumerate(features):
        linhas.append({"feature": f, "coef_padronizado": round(float(m.coef_[i]), 4),
                       "ic95_low": round(float(m.ci_[i][0]), 4),
                       "ic95_high": round(float(m.ci_[i][1]), 4),
                       "p_valor": round(float(m.pvals_[i]), 4),
                       "sinal_esperado": SINAL_ESPERADO.get(f),
                       "sinal_bate": bool(np.sign(m.coef_[i]) == SINAL_ESPERADO.get(f, 0)),
                       "ic_cruza_zero": bool(m.ci_[i][0] <= 0 <= m.ci_[i][1])})
    return pd.DataFrame(linhas), {"n_usado": int(len(dd)), "n_pos": int((y == 1).sum()),
                                  "n_neg": int((y == 0).sum()), "loglik": float(m.loglik_)}


def loo_auc(d: pd.DataFrame, features: list[str]) -> dict:
    dd = d.dropna(subset=features + ["label"]).reset_index(drop=True)
    X = dd[features].to_numpy(dtype="float64")
    y = dd["label"].astype(int).to_numpy()
    yt, ys = [], []
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        ys.append(clf.predict_proba(Xte)[:, 1][0])
        yt.append(y[te][0])
    return {"n_usado": int(len(dd)), "loo_auc": round(float(roc_auc_score(yt, ys)), 4)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===MOD_RECIFE03_PLUVIAL_FONTE_UNICA===")

    if not HARM.exists():
        print(f"ABORTADO: {HARM} ausente.")
        return 1

    d = pd.read_csv(HARM, low_memory=False)
    d = d.rename(columns={"label": "label"})
    print(f"ENTRADA n={len(d)} ({int((d.label==1).sum())} pos / "
          f"{int((d.label==0).sum())} neg)")
    print(f"fonte_chuva (deve ser unica): "
          f"{d['rain_data_source'].value_counts().to_dict()}")

    resid, ortho_report = orthogonalizar(d["rain_max_24h_chirps"],
                                         d["rain_decay_index_api_chirps"])
    d["rain_peak_residual_orthogonalized"] = resid
    print(f"\n[1] Ortogonalizacao GLOBAL (fonte unica, sem mais 'por fonte'): "
          f"{json.dumps(ortho_report)}")

    print("\n[2] Firth multivariado, 6 features (terreno NATIVO, igual ao v12)")
    firth_df, firth_report = firth_multivariado(d, FEATURES)
    print(json.dumps(firth_report, indent=2))
    print(firth_df.to_string(index=False))

    print("\n[3] LOO-AUC")
    auc_report = loo_auc(d, FEATURES)
    print(json.dumps(auc_report, indent=2))

    print("\n[4] COMPARACAO COM O V12 PUBLICADO (mesma metodologia, so a fonte de chuva muda)")
    mapa_v12 = {"elevation_m_original": "elevation_m", "slope_deg_original": "slope_deg",
               "hand_m_dinf_original": "hand_m_dinf", "twi_dinf_original": "twi_dinf",
               "rain_peak_residual_orthogonalized": "rain_peak_residual_orthogonalized",
               "rain_decay_index_api_chirps": "rain_decay_index_api_chirps"}
    comp = {
        "loo_auc": {"v12_publicado": V12_PUBLICADO["loo_auc"],
                    "fonte_unica_agora": auc_report["loo_auc"],
                    "delta": round(auc_report["loo_auc"] - V12_PUBLICADO["loo_auc"], 4)},
        "n": {"v12_publicado": V12_PUBLICADO["n"], "fonte_unica_agora": firth_report["n_usado"]},
        "coeficientes": {},
    }
    for f_novo, f_v12 in mapa_v12.items():
        row = firth_df.loc[firth_df.feature == f_novo].iloc[0]
        pub = V12_PUBLICADO["coef"][f_v12]
        comp["coeficientes"][f_v12] = {
            "v12_publicado": {"coef": pub["coef"], "p": pub["p"]},
            "fonte_unica_agora": {"coef": float(row.coef_padronizado),
                                  "p": float(row.p_valor)},
        }
    print(json.dumps(comp, indent=2))

    firth_df.to_csv(OUT / "firth_multivariado_fonte_unica.csv", index=False)
    (OUT / "resumo.json").write_text(json.dumps({
        "entrada": {"n": int(len(d)), "pos": int((d.label==1).sum()),
                    "neg": int((d.label==0).sum())},
        "fonte_chuva": d["rain_data_source"].value_counts().to_dict(),
        "ortogonalizacao": ortho_report,
        "firth": firth_report, "loo": auc_report,
        "comparacao_com_v12_publicado": comp,
        "terreno": "NATIVO (elevation_m_original etc, igual ao v12 -- resolucao "
                   "de terreno NAO mudou neste script, so a fonte de chuva)",
        "seed": SEED,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
