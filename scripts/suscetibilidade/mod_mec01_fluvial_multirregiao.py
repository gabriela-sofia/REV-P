"""
MOD-MEC-01 -- O primeiro modelo multirregiao com cadeia unica e mecanismo unico.

O QUE MUDOU PARA ESTE MODELO SER POSSIVEL:
tres correcoes tiveram que acontecer antes, e cada uma sozinha invalidaria o
resultado.

  1. ANALOGIA POR AOI. O ranking anterior media relevo no centroide da
     ativacao; cinco dos seis "analogos de Petropolis" eram planicies.
  2. CADEIA UNICA. Recife derivava HAND com WhiteboxTools; os analogos BAIXAVAM
     o produto global. E o limiar de canal, declarado em percentil, valia
     0,1123 km2 em Recife e 0,7507 em Petropolis -- 6,7 vezes mais esparso.
  3. MECANISMO UNICO. Recife e pluvial urbano (HAND com p=0,978, ausente); o
     conjunto externo e fluvial. Somar os dois concilia fenomenos diferentes.

Este script roda sobre o que sobrou depois das tres: **so mecanismo fluvial, so
cadeia WhiteboxTools 30 m com limiar de 0,1123 km2**.

A COMPOSICAO, e por que ela e interessante:
o subconjunto une AOIs de serra tropical (Equador, Haiti, Sri Lanka, Honduras,
La Reunion, Mocambique) com Curitiba -- fluvial urbano em planalto subtropical.
Sao contextos muito distintos sob o mesmo mecanismo e a mesma medida. Se o
modelo se sustentar com validacao agrupada, e evidencia de que a relacao
terreno-inundacao fluvial e transferivel entre climas.

O TESTE POR FONTE (leave-one-source-out) e mais duro que o GroupKFold: treina
em todas as fontes menos uma e testa na que ficou de fora. Responde "o modelo
funciona onde nunca viu nada" em vez de "funciona em evento que nao viu".

CRITERIOS, de ext_criterios_de_acerto_v1.md, fixados antes:
    AUC com CV agrupada em 0,70-0,88; acima de 0,95 e suspeita de vazamento
    hand_m negativo e twi_dinf positivo obrigatorios
    gap treino-validacao abaixo de 0,15; EPV minimo de 10 em GRUPOS

NAO faz: nao ajusta hiperparametro, nao seleciona feature por desempenho, nao
mistura cadeia global com harmonizada.

Uso:
    python scripts/suscetibilidade/mod_mec01_fluvial_multirregiao.py
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
from mod_serra01_ingreme_2features import (  # noqa: E402
    EPV_MINIMO, N_BOOT, SEMENTE, AUC_MIN, AUC_MAX, AUC_SUSPEITA, GAP_MAX,
    ajustar, padronizar,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ds-02-mecanismo" / "dataset_fluvial_enxurrada.csv"
OUT = RUNS / "mod-mec-01"

FEATURES = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
SINAL_EXIGIDO = {"hand_m": -1, "twi_dinf": +1}
N_FOLDS = 5


def main() -> int:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("===MOD-MEC-01: FLUVIAL MULTIRREGIAO, CADEIA UNICA===")

    if not ENTRADA.exists():
        print(f"ABORTADO: {ENTRADA} ausente. Rode ds02 antes.")
        return 1
    d = pd.read_csv(ENTRADA, low_memory=False)

    # SO cadeia harmonizada. Misturar com a global e o erro que este script existe
    # para nao cometer -- as duas produzem HAND com definicoes diferentes de canal.
    antes = len(d)
    d = d[d.cadeia == "wbt30"].dropna(subset=FEATURES + ["classe", "grupo_cv"])
    d = d[d.classe.isin([0, 1])].reset_index(drop=True)
    print(f"PONTOS={len(d):,} (descartados {antes-len(d):,}: cadeia global ou feature ausente)")
    print(f"  pos={int(d.classe.sum()):,} neg={len(d)-int(d.classe.sum()):,} "
          f"grupos={d.grupo_cv.nunique():,} fontes={d.fonte.nunique()}")
    print("--- composicao por fonte ---")
    comp = (d.groupby("fonte").agg(n=("classe", "size"), pos=("classe", "sum"),
                                   grupos=("grupo_cv", "nunique"))
              .assign(neg=lambda x: x.n - x.pos).sort_values("n", ascending=False))
    print(comp[["n", "pos", "neg", "grupos"]].to_string())

    epv = d.grupo_cv.nunique() / len(FEATURES)
    print(f"EPV={epv:.1f} (minimo {EPV_MINIMO})")
    if epv < EPV_MINIMO:
        print("ABORTADO: EPV abaixo do minimo.")
        return 3

    y = d.classe.to_numpy().astype(int)
    X = d[FEATURES].to_numpy(dtype=float)
    g = d.grupo_cv.to_numpy()
    Xz, mu, sd = padronizar(X)

    # ---- A. GroupKFold por grupo ----
    print("---A. VALIDACAO AGRUPADA (grupo = evento/AOI/unidade)---")
    aucs, aucs_tr = [], []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(Xz, y, groups=g):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        m = ajustar(Xz[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(Xz[te])[:, 1]))
        aucs_tr.append(roc_auc_score(y[tr], m.predict_proba(Xz[tr])[:, 1]))
    auc = float(np.mean(aucs))
    gap = float(np.mean(aucs_tr) - auc)
    print(f"  AUC_CV={auc:.4f} (folds={len(aucs)}, min={min(aucs):.4f}, "
          f"max={max(aucs):.4f}) gap={gap:+.4f}")

    # ---- B. coeficientes com IC por bootstrap de grupos ----
    print("---B. COEFICIENTES (IC por bootstrap de GRUPOS)---")
    m_full = ajustar(Xz, y)
    coef = m_full.coef_.ravel()[:len(FEATURES)]
    grupos = d.grupo_cv.unique()
    idx = {gr: d.index[d.grupo_cv == gr].to_numpy() for gr in grupos}
    boots = []
    for _ in range(N_BOOT):
        esc = rng.choice(grupos, size=len(grupos), replace=True)
        linhas = np.concatenate([idx[gr] for gr in esc])
        s = d.loc[linhas]
        ys = s.classe.to_numpy().astype(int)
        if len(np.unique(ys)) < 2:
            continue
        Xs, _, _ = padronizar(s[FEATURES].to_numpy(dtype=float))
        try:
            boots.append(ajustar(Xs, ys).coef_.ravel()[:len(FEATURES)])
        except Exception:  # noqa: BLE001
            continue
    boots = np.array(boots)
    ic = (np.percentile(boots, [2.5, 97.5], axis=0) if len(boots) > 30
          else np.full((2, len(FEATURES)), np.nan))
    for i, f in enumerate(FEATURES):
        lo, hi = ic[0, i], ic[1, i]
        marca = "" if (np.isnan(lo) or lo * hi > 0) else "  <-- IC CRUZA ZERO"
        print(f"  {f:14s} coef={coef[i]:+8.4f}  IC95=[{lo:+7.4f},{hi:+7.4f}]{marca}")

    # ---- C. leave-one-source-out ----
    print("---C. LEAVE-ONE-SOURCE-OUT (o teste mais duro)---")
    loso = []
    for fonte in sorted(d.fonte.unique()):
        tr_m = (d.fonte != fonte).to_numpy()
        te_m = ~tr_m
        if len(np.unique(y[te_m])) < 2 or len(np.unique(y[tr_m])) < 2:
            continue
        mu2, sd2 = X[tr_m].mean(0), X[tr_m].std(0)
        sd2 = np.where(sd2 == 0, 1, sd2)
        mm = ajustar((X[tr_m] - mu2) / sd2, y[tr_m])
        a = float(roc_auc_score(y[te_m], mm.predict_proba((X[te_m] - mu2) / sd2)[:, 1]))
        loso.append({"fonte_excluida": fonte, "n_teste": int(te_m.sum()),
                     "grupos_teste": int(d.loc[te_m, "grupo_cv"].nunique()),
                     "auc": round(a, 4)})
        print(f"  sem {fonte:22s} -> AUC={a:.4f} (n={int(te_m.sum()):,})")

    # ---- veredito ----
    print("---VEREDITO---")
    falhas = []
    if auc >= AUC_SUSPEITA:
        falhas.append(f"AUC {auc:.4f} >= {AUC_SUSPEITA} (vazamento)")
    elif not (AUC_MIN <= auc <= AUC_MAX):
        falhas.append(f"AUC {auc:.4f} fora de [{AUC_MIN}, {AUC_MAX}]")
    if abs(gap) > GAP_MAX:
        falhas.append(f"gap {gap:+.4f} acima de {GAP_MAX}")
    for f, exigido in SINAL_EXIGIDO.items():
        i = FEATURES.index(f)
        if np.sign(coef[i]) != exigido:
            falhas.append(f"{f} com sinal invertido")
        if not np.isnan(ic[0, i]) and ic[0, i] * ic[1, i] <= 0:
            falhas.append(f"{f} com IC95 cruzando zero")
    for x in falhas:
        print(f"  REPROVA: {x}")
    veredito = "COERENTE_COM_CRITERIOS" if not falhas else "FORA_DOS_CRITERIOS"
    print(f"  VEREDITO={veredito}")

    (OUT / "resultado.json").write_text(json.dumps({
        "mecanismo": "FLUVIAL_ENXURRADA", "cadeia": "wbt30",
        "features": FEATURES, "n": int(len(d)),
        "grupos": int(d.grupo_cv.nunique()), "fontes": sorted(d.fonte.unique()),
        "epv": round(epv, 1),
        "composicao": json.loads(comp.to_json(orient="index")),
        "auc_cv": round(auc, 4), "gap": round(gap, 4), "folds": len(aucs),
        "coef": {f: round(float(c), 4) for f, c in zip(FEATURES, coef)},
        "ic95": {f: [round(float(ic[0, i]), 4), round(float(ic[1, i]), 4)]
                 for i, f in enumerate(FEATURES)},
        "leave_one_source_out": loso,
        "falhas": falhas, "veredito": veredito,
        "semente": SEMENTE, "n_boot": N_BOOT,
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
