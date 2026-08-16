"""
MOD-RECIFE-04 -- Um ponto real, com escore real, para a figura de exemplo do
Documento de Planejamento (nao entra em nenhuma decisao de modelo).

Reusa a MESMA base e o MESMO ajuste de `mod_recife03_pluvial_fonte_unica.py`
(fonte de chuva unica, terreno nativo, 6 features causais). O que este script
faz que aquele nao fazia: guarda a previsao por ponto (o `loo_auc` original so
agregava em AUC) e escolhe UM ponto real -- nao fictício -- para ilustrar o
que o contrato de inferencia devolveria.

CRITERIO DE ESCOLHA, declarado para nao parecer garimpo de exemplo bonito:
entre os positivos verdadeiros (rotulo = enchente registrada), o ponto cuja
probabilidade prevista fora da amostra (LOO) fica mais perto da MEDIANA dos
positivos verdadeiros -- nao o mais extremo, o mais tipico.

PRIVACIDADE: a base tem lat/lon do ponto (endereco aproximado de ocorrencia
registrada pela Defesa Civil). Este script publica o BAIRRO, nao a
coordenada -- e o mesmo nivel de granularidade que a propria Defesa Civil usa
em boletim publico, e evita apontar um endereco especifico numa figura de
TCC.

IC do escore do ponto escolhido: bootstrap de 1000 reamostragens dos OUTROS
268 pontos, reajustando e prevendo sempre no MESMO ponto fixo -- e o desvio
do modelo diante de diferentes amostras de treino, nao incerteza de medida.

NAO faz: nao promove esta rota a canonica, nao altera mod_recife03, nao entra
na tabela unica nem no pool fluvial.

Uso:
    python scripts/suscetibilidade/mod_recife04_exemplo_para_figura.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
HARM = REPO / "local_runs" / "ter-03-brasil-harmonizado" / "recife_harmonizado.csv"
OUT = REPO / "local_runs" / "mod-recife-04-exemplo-figura"
SEED = 20260723

FEATURES = ["elevation_m_original", "slope_deg_original", "hand_m_dinf_original",
            "twi_dinf_original", "rain_peak_residual_orthogonalized",
            "rain_decay_index_api_chirps"]
N_BOOT = 1000


def orthogonalizar(rain_max: pd.Series, decay: pd.Series) -> pd.Series:
    x = decay.to_numpy(dtype="float64")
    y = rain_max.to_numpy(dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    xm = x[ok].mean()
    beta = float(((x[ok] - xm) @ y[ok]) / ((x[ok] - xm) @ (x[ok] - xm)))
    intercept = float(y[ok].mean() - beta * xm)
    resid = pd.Series(np.nan, index=rain_max.index)
    resid.iloc[np.where(ok)[0]] = y[ok] - (intercept + beta * x[ok])
    return resid


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===MOD_RECIFE04_EXEMPLO_PARA_FIGURA===")

    d = pd.read_csv(HARM, low_memory=False)
    d["rain_peak_residual_orthogonalized"] = orthogonalizar(
        d["rain_max_24h_chirps"], d["rain_decay_index_api_chirps"])

    dd = d.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    X = dd[FEATURES].to_numpy(dtype="float64")
    y = dd["label"].astype(int).to_numpy()
    print(f"ENTRADA n={len(dd)} ({int((y==1).sum())} pos / {int((y==0).sum())} neg)")

    # ---- LOO: previsao real fora da amostra, para cada ponto ----
    proba_loo = np.full(len(dd), np.nan)
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        proba_loo[te[0]] = clf.predict_proba(Xte)[:, 1][0]
    dd["proba_loo"] = proba_loo

    # ---- escolhe o positivo verdadeiro mais TIPICO, nao o mais extremo ----
    pos = dd[dd.label == 1].copy()
    mediana_pos = pos["proba_loo"].median()
    pos["dist_mediana"] = (pos["proba_loo"] - mediana_pos).abs()
    alvo = pos.sort_values("dist_mediana").iloc[0]
    idx_alvo = int(dd.index.get_loc(alvo.name))
    print(f"\nESCOLHIDO: {alvo.point_id}  bairro={alvo.neighborhood}  "
          f"proba_loo={alvo.proba_loo:.4f}  (mediana dos positivos={mediana_pos:.4f})")

    # ---- bootstrap do escore NESTE ponto fixo, reamostrando o resto ----
    outros = np.array([i for i in range(len(dd)) if i != idx_alvo])
    rng = np.random.default_rng(SEED)
    x_alvo = X[idx_alvo:idx_alvo + 1]
    boot = []
    for _ in range(N_BOOT):
        bi = rng.choice(outros, size=len(outros), replace=True)
        sc = StandardScaler()
        Xb = sc.fit_transform(X[bi])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xb, y[bi])
        boot.append(float(clf.predict_proba(sc.transform(x_alvo))[:, 1][0]))
    boot = np.array(boot)
    ic_low, ic_high = np.percentile(boot, [2.5, 97.5])
    print(f"IC95% bootstrap (N={N_BOOT}, reamostrando os outros {len(outros)} "
          f"pontos): [{ic_low:.4f}, {ic_high:.4f}]")

    resultado = {
        "point_id": str(alvo.point_id),
        "bairro": str(alvo.neighborhood),
        "label": int(alvo.label),
        "event_date": str(alvo.get("data_evento", alvo.get("event_date", ""))),
        "proba_loo": round(float(alvo.proba_loo), 4),
        "ic95": [round(float(ic_low), 4), round(float(ic_high), 4)],
        "n_boot": N_BOOT,
        "seed": SEED,
        "criterio_escolha": "positivo verdadeiro mais proximo da mediana dos "
                            "positivos por proba_loo, nao o mais extremo",
        "variaveis_do_ponto": {f: round(float(alvo[f]), 4) for f in FEATURES},
        "rain_data_source": str(d.loc[d.point_id == alvo.point_id, "rain_data_source"].iloc[0]),
        "nota_privacidade": "bairro publicado, nao lat/lon -- mesma granularidade "
                            "que boletim publico da Defesa Civil",
    }
    (OUT / "exemplo.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGRAVADO={OUT}/exemplo.json")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
