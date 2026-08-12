"""
MOD-PROSP-01 -- Validacao prospectiva: treinar no passado, testar no futuro.

POR QUE ESTE E O TESTE QUE FALTAVA:
toda validacao do projeto ate agora foi por GRUPO (evento ou AOI). Isso trata
autocorrelacao espacial, mas nao responde a pergunta operacional: **um modelo
ajustado com o que ja aconteceu acerta o que ainda vai acontecer?** Um modelo de
suscetibilidade que so funciona quando ve eventos contemporaneos aos de treino
nao serve para prevenir nada.

E o teste que ataca diretamente o colapso temporal de Curitiba (AUC 0,6459 ->
0,5246 em holdout 2026) que sete diagnosticos internos nao explicaram. Se o
mesmo colapso aparecer na Inglaterra, e propriedade do metodo. Se nao aparecer,
e propriedade dos dados de Curitiba.

O DADO: 401 eventos ingleses entre 2000 e 2025. A data vem do positivo; o
negativo herda a data do seu proprio evento (mesmo `grupo_cv`), porque foi
amostrado para aquele evento. Sem essa heranca, so 428 dos 3.738 negativos
teriam data e o holdout ficaria desbalanceado por construcao. Com ela: 328
grupos, 3.738 positivos e 2.962 negativos.

O DESENHO -- janela expansiva (walk-forward), nao divisao unica:
para cada ponto de corte, treina em TODOS os eventos anteriores e testa em uma
janela de eventos posteriores. Divisao unica daria um numero que depende do
corte escolhido; a janela expansiva mostra a TRAJETORIA, que e o que distingue
degradacao real de azar num corte.

TRAVA DE EPV: um fold so e avaliado se o treino tiver pelo menos
10 x n_features GRUPOS. Com 4 features isso exige 40 eventos de treino, o que
descarta os cortes mais antigos -- corretamente, porque ali nao ha amostra.

O QUE SIGNIFICA CADA RESULTADO:
  AUC estavel ao longo dos cortes    -> a relacao terreno-inundacao nao caduca;
                                        o modelo e prospectivamente util
  AUC caindo de forma monotonica     -> o modelo aprende regime, nao fisica
  AUC oscilando sem tendencia        -> variancia de amostra pequena, nao
                                        degradacao; nao confundir os dois

CRITERIO declarado antes: comparar cada fold prospectivo contra a faixa de
ext_criterios_de_acerto_v1.md (0,70-0,88). Queda abaixo de 0,60 em fold com
EPV suficiente e sinal de degradacao real.

NAO faz: nao ajusta hiperparametro, nao escolhe o corte pelo resultado, nao
declara modelo final.

Uso:
    python scripts/suscetibilidade/mod_prosp01_holdout_temporal.py
    python scripts/suscetibilidade/mod_prosp01_holdout_temporal.py --6features
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import susc_firth_shim  # noqa: F401,E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
DATASET = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
OUT = RUNS / "mod-prosp-01"

TOPO = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
CHUVA = ["rain_max_24h", "rain_decay_index_api"]
EPV_MINIMO = 10
MIN_GRUPOS_TESTE = 8
SEMENTE = 20260812

AUC_MIN, AUC_MAX = 0.70, 0.88
AUC_DEGRADACAO = 0.60


def ajustar(X, y):
    from firthlogist import FirthLogisticRegression

    m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
    m.fit(X, y)
    return m


def main() -> int:
    from sklearn.metrics import roc_auc_score

    features = TOPO + CHUVA if "--6features" in sys.argv else TOPO
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===MOD-PROSP-01: HOLDOUT TEMPORAL===")
    print(f"FEATURES={features}")

    if not DATASET.exists():
        print(f"ABORTADO: {DATASET} ausente.")
        return 1
    d = pd.read_csv(DATASET, low_memory=False)
    uk = d[(d.regiao == "UK_noroeste") & d.classe.isin([0, 1])].copy()
    uk["data"] = pd.to_datetime(uk.data_evento, errors="coerce")

    # heranca da data pelo grupo -- ver cabecalho
    uk["data_grupo"] = uk.groupby("grupo_cv")["data"].transform(
        lambda s: s.dropna().min() if s.notna().any() else pd.NaT)
    antes = len(uk)
    uk = uk.dropna(subset=["data_grupo"] + features)
    print(f"PONTOS={len(uk):,} (descartados {antes-len(uk):,} sem data ou sem feature) "
          f"grupos={uk.grupo_cv.nunique()} pos={int(uk.classe.sum()):,} "
          f"neg={len(uk)-int(uk.classe.sum()):,}")

    datas = (uk.groupby("grupo_cv")["data_grupo"].min().sort_values())
    ordem = datas.index.tolist()
    print(f"PERIODO={datas.min().date()} -> {datas.max().date()} "
          f"grupos_ordenados={len(ordem)}")

    min_treino = EPV_MINIMO * len(features)
    print(f"EPV_MINIMO={EPV_MINIMO} -> treino precisa de >= {min_treino} grupos")

    janela = max(MIN_GRUPOS_TESTE, len(ordem) // 10)
    linhas = []
    i = min_treino
    while i + MIN_GRUPOS_TESTE <= len(ordem):
        g_tr = set(ordem[:i])
        g_te = set(ordem[i:i + janela])
        tr = uk[uk.grupo_cv.isin(g_tr)]
        te = uk[uk.grupo_cv.isin(g_te)]
        if te.classe.nunique() < 2 or tr.classe.nunique() < 2:
            i += janela
            continue

        Xtr = tr[features].to_numpy(dtype=float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd == 0, 1, sd)
        m = ajustar((Xtr - mu) / sd, tr.classe.to_numpy().astype(int))
        p_te = m.predict_proba((te[features].to_numpy(dtype=float) - mu) / sd)[:, 1]
        p_tr = m.predict_proba((Xtr - mu) / sd)[:, 1]
        auc_te = float(roc_auc_score(te.classe.to_numpy().astype(int), p_te))
        auc_tr = float(roc_auc_score(tr.classe.to_numpy().astype(int), p_tr))
        corte = datas.loc[ordem[i]]

        linhas.append({
            "corte": str(corte.date()), "grupos_treino": len(g_tr),
            "grupos_teste": len(g_te), "n_treino": len(tr), "n_teste": len(te),
            "epv_treino": round(len(g_tr) / len(features), 1),
            "auc_prospectivo": round(auc_te, 4), "auc_treino": round(auc_tr, 4),
            "gap": round(auc_tr - auc_te, 4),
            "coef": {f: round(float(c), 4)
                     for f, c in zip(features, m.coef_.ravel()[:len(features)])},
        })
        print(f"  corte={str(corte.date()):12s} treino={len(g_tr):3d}g "
              f"teste={len(g_te):3d}g  AUC_prosp={auc_te:.4f}  "
              f"(treino {auc_tr:.4f}, gap {auc_tr-auc_te:+.4f})")
        i += janela

    if not linhas:
        print("ABORTADO: nenhum fold com EPV suficiente.")
        return 2

    r = pd.DataFrame(linhas)
    r.to_csv(OUT / "folds.csv", index=False)
    a = r.auc_prospectivo.to_numpy()
    print("\n---RESUMO---")
    print(f"  folds={len(a)}  AUC medio={a.mean():.4f}  mediana={np.median(a):.4f}  "
          f"min={a.min():.4f}  max={a.max():.4f}  desvio={a.std():.4f}")

    # tendencia: correlacao entre ordem do corte e AUC
    tend = float(np.corrcoef(np.arange(len(a)), a)[0, 1]) if len(a) > 2 else float("nan")
    print(f"  tendencia (corr entre ordem do corte e AUC) = {tend:+.3f}")

    degradou = int((a < AUC_DEGRADACAO).sum())
    na_faixa = int(((a >= AUC_MIN) & (a <= AUC_MAX)).sum())
    print(f"  folds na faixa [{AUC_MIN}, {AUC_MAX}] = {na_faixa}/{len(a)}")
    print(f"  folds abaixo de {AUC_DEGRADACAO} = {degradou}/{len(a)}")

    if tend < -0.5 and degradou > 0:
        veredito = "DEGRADACAO_TEMPORAL"
    elif a.mean() >= AUC_MIN and degradou == 0:
        veredito = "PROSPECTIVAMENTE_ESTAVEL"
    else:
        veredito = "INCONCLUSIVO_VARIANCIA_ALTA"
    print(f"  VEREDITO={veredito}")

    (OUT / "resultado.json").write_text(json.dumps({
        "features": features, "n": int(len(uk)), "grupos": int(uk.grupo_cv.nunique()),
        "periodo": [str(datas.min().date()), str(datas.max().date())],
        "epv_minimo": EPV_MINIMO, "janela_teste_grupos": janela,
        "folds": linhas,
        "auc_medio": round(float(a.mean()), 4),
        "auc_mediana": round(float(np.median(a)), 4),
        "auc_min": round(float(a.min()), 4), "auc_max": round(float(a.max()), 4),
        "desvio": round(float(a.std()), 4), "tendencia": round(tend, 3),
        "folds_na_faixa": na_faixa, "folds_degradados": degradou,
        "veredito": veredito, "semente": SEMENTE,
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
