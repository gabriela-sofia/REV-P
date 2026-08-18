"""
MOD-SERRA-01 -- O modelo de encosta: HAND e TWI em terreno ingreme.

A PERGUNTA:
o REV-P quer propagar para Petropolis a partir de regioes analogas. Ate
10/08/2026 o conjunto de treino nao tinha terreno de serra: cinco dos seis
"analogos" eram planicies de inundacao, selecionados por um relevo lido no
CENTROIDE DA ATIVACAO em vez das AOIs. Corrigido isso e baixadas sete
ativacoes novas, ha 22 AOIs que passam no criterio de serra.

A pergunta que este script responde: **a relacao HAND -> inundacao se comporta
em encosta como se comporta em planicie?** Se sim, propagar para Petropolis
tem base. Se nao, a propagacao precisa de modelo proprio.

POR QUE DUAS FEATURES, E NAO QUATRO:
com 22 grupos, quatro features dao EPV 5,5 -- abaixo do minimo de 10 do
projeto, e o proprio projeto proibe interpretar modelo nessa condicao. Duas
features dao EPV 11.

A escolha de QUAIS duas nao e por desempenho, e sim pelo que ja foi medido:
`elevation_m` e `slope_deg` INVERTEM DE SINAL entre fontes (ver
ext_o_que_nao_e_enchente_v2.md, secao 5: concordancia de sinal 1 de 3, so
`hand_m` concorda). Isso e a assinatura de descritor de contexto local, nao de
processo. `hand_m` e `twi_dinf` sao os dois termos causais do balanco:
quanto precisa subir para alcancar, e para onde a agua converge.

Reduzir a essas duas e consequencia do que foi aprendido, nao conveniencia
amostral. Se fosse conveniencia, a escolha teria sido pelas duas de maior AUC.

O TESTE QUE IMPORTA -- tres partes:

  A. CARACTERIZACAO. Como HAND e TWI se distribuem em encosta contra planicie?
     Em serra o positivo deve estar em HAND muito baixo (fundo de vale
     confinado) e o negativo pode estar a dezenas de metros acima, porque a
     encosta sobe rapido. Em planicie a separacao e menor em metros.

  B. MODELO NA SERRA. Firth, GroupKFold por AOI, IC por bootstrap que
     reamostra GRUPOS e nao linhas (regra U1/U2/U3 -- reamostrar linhas
     infla a precisao por pseudo-replicacao).

  C. TRANSFERENCIA CRUZADA. Modelo da serra avaliado na planicie e vice-versa.
     E o teste mais duro de "a fisica e a mesma".

CRITERIOS DE ACEITACAO, fixados em ext_criterios_de_acerto_v1.md ANTES desta
rodada e nao ajustaveis por resultado:
    AUC com CV agrupada em 0,70-0,88   (acima de 0,95 = suspeita de vazamento)
    hand_m com sinal NEGATIVO          (obrigatorio, independente do AUC)
    twi_dinf com sinal POSITIVO        (obrigatorio)
    gap treino-validacao abaixo de 0,15
    nenhum fold com AUC >= 0,999

NAO faz: nao ajusta hiperparametro, nao seleciona feature por desempenho, nao
declara modelo final, nao toca em Petropolis.

Uso:
    python scripts/suscetibilidade/mod_serra01_ingreme_2features.py
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
RELEVO = RUNS / "cems-02-analogos-v2" / "verificacao_relevo_por_aoi.csv"
OUT = RUNS / "mod-serra-01"

FEATURES = ["hand_m", "twi_dinf"]
EPV_MINIMO = 10
N_FOLDS = 5
N_BOOT = 600
SEMENTE = 20260811

# Faixas declaradas em ext_criterios_de_acerto_v1.md
AUC_MIN, AUC_MAX = 0.70, 0.88
AUC_SUSPEITA = 0.95
GAP_MAX = 0.15
SINAL_EXIGIDO = {"hand_m": -1, "twi_dinf": +1}


def ajustar(X, y):
    from firthlogist import FirthLogisticRegression

    m = FirthLogisticRegression(max_iter=500, skip_pvals=True, skip_ci=True)
    m.fit(X, y)
    return m


def padronizar(X):
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd == 0, 1, sd), mu, np.where(sd == 0, 1, sd)


def avaliar_cv(sub: pd.DataFrame):
    """AUC por GroupKFold + gap treino-validacao. Grupo = AOI."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    y = sub["classe"].to_numpy().astype(int)
    X = sub[FEATURES].to_numpy(dtype=float)
    g = sub["grupo_cv"].to_numpy()
    Xz, _, _ = padronizar(X)

    folds = min(N_FOLDS, len(np.unique(g)))
    aucs_val, aucs_tr = [], []
    for tr, te in GroupKFold(n_splits=folds).split(Xz, y, groups=g):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        m = ajustar(Xz[tr], y[tr])
        aucs_val.append(roc_auc_score(y[te], m.predict_proba(Xz[te])[:, 1]))
        aucs_tr.append(roc_auc_score(y[tr], m.predict_proba(Xz[tr])[:, 1]))
    return np.array(aucs_val), np.array(aucs_tr)


def bootstrap_por_grupo(sub: pd.DataFrame, rng):
    """IC reamostrando GRUPOS. Reamostrar LINHAS infla a precisao -- regra U2."""
    grupos = sub["grupo_cv"].unique()
    idx_por_grupo = {gr: sub.index[sub.grupo_cv == gr].to_numpy() for gr in grupos}
    coefs = []
    for _ in range(N_BOOT):
        esc = rng.choice(grupos, size=len(grupos), replace=True)
        linhas = np.concatenate([idx_por_grupo[gr] for gr in esc])
        s = sub.loc[linhas]
        y = s["classe"].to_numpy().astype(int)
        if len(np.unique(y)) < 2:
            continue
        Xz, _, _ = padronizar(s[FEATURES].to_numpy(dtype=float))
        try:
            coefs.append(ajustar(Xz, y).coef_.ravel()[:len(FEATURES)])
        except Exception:  # noqa: BLE001
            continue
    return np.array(coefs)


def main() -> int:
    from sklearn.metrics import roc_auc_score

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("===MOD-SERRA-01: HAND E TWI EM TERRENO INGREME===")

    for p in (DATASET, RELEVO):
        if not p.exists():
            print(f"ABORTADO: {p} ausente.")
            return 1

    rel = pd.read_csv(RELEVO)
    ingremes = set(rel.loc[rel.analogo_de_petropolis, "aoi"])
    if not ingremes:
        print("ABORTADO: nenhuma AOI aprovada como ingreme. Rode cems03 --verificar.")
        return 1

    df = pd.read_csv(DATASET, low_memory=False)
    df = df[df.classe.isin([0, 1])].dropna(subset=FEATURES).reset_index(drop=True)
    df = df[df.regiao.str.startswith("analogo_")]

    serra = df[df.grupo_cv.isin(ingremes)].reset_index(drop=True)
    plano = df[~df.grupo_cv.isin(ingremes)].reset_index(drop=True)

    for nome, s in (("SERRA", serra), ("PLANICIE", plano)):
        print(f"{nome:9s} n={len(s):6,} pos={int(s.classe.sum()):5,} "
              f"neg={len(s)-int(s.classe.sum()):5,} grupos={s.grupo_cv.nunique():3d}")

    # --- trava de EPV: mesma regra de todos os scripts do projeto ---
    for nome, s in (("serra", serra), ("planicie", plano)):
        epv = s.grupo_cv.nunique() / len(FEATURES)
        print(f"  EPV_{nome}={epv:.1f} (grupos/features, minimo={EPV_MINIMO})")
        if epv < EPV_MINIMO:
            print(f"ABORTADO: EPV de '{nome}' abaixo do minimo. Nao vou interpretar "
                  f"modelo com suficiencia amostral incompativel com a regra do projeto.")
            return 3

    # ---------------- A. caracterizacao ----------------
    print("---A. COMO HAND E TWI SE DISTRIBUEM---")
    carac = {}
    for nome, s in (("serra", serra), ("planicie", plano)):
        d = {}
        for f in FEATURES:
            pos = float(s.loc[s.classe == 1, f].median())
            neg = float(s.loc[s.classe == 0, f].median())
            d[f] = {"pos": round(pos, 3), "neg": round(neg, 3),
                    "contraste": round(neg - pos, 3)}
            print(f"  {nome:9s} {f:10s} pos={pos:8.2f} neg={neg:8.2f} "
                  f"contraste={neg-pos:+8.2f}")
        carac[nome] = d

    # ---------------- B. modelo na serra ----------------
    print("---B. MODELO POR SUBCONJUNTO---")
    modelos, resultados = {}, {}
    for nome, s in (("serra", serra), ("planicie", plano)):
        val, tr = avaliar_cv(s)
        Xz, mu, sd = padronizar(s[FEATURES].to_numpy(dtype=float))
        m = ajustar(Xz, s["classe"].to_numpy().astype(int))
        coef = m.coef_.ravel()[:len(FEATURES)]
        modelos[nome] = (m, mu, sd)

        boot = bootstrap_por_grupo(s, rng)
        ic = (np.percentile(boot, [2.5, 97.5], axis=0)
              if len(boot) > 30 else np.full((2, len(FEATURES)), np.nan))

        auc = float(np.mean(val)) if len(val) else float("nan")
        gap = float(np.mean(tr) - np.mean(val)) if len(val) else float("nan")
        resultados[nome] = {"auc_cv": round(auc, 4), "gap": round(gap, 4),
                            "folds": len(val), "auc_max_fold": round(float(val.max()), 4) if len(val) else None,
                            "n": len(s), "grupos": int(s.grupo_cv.nunique()),
                            "boot_validos": len(boot),
                            "coef": {f: round(float(c), 4) for f, c in zip(FEATURES, coef)},
                            "ic95": {f: [round(float(ic[0, i]), 4), round(float(ic[1, i]), 4)]
                                     for i, f in enumerate(FEATURES)}}
        print(f"  {nome:9s} AUC_CV={auc:.4f} (folds={len(val)}) gap={gap:+.4f}")
        for i, f in enumerate(FEATURES):
            lo, hi = ic[0, i], ic[1, i]
            zero = "" if (np.isnan(lo) or lo * hi > 0) else "  <-- IC CRUZA ZERO"
            print(f"      {f:10s} coef={coef[i]:+8.4f}  IC95=[{lo:+7.4f},{hi:+7.4f}]{zero}")

    # ---------------- C. transferencia cruzada ----------------
    print("---C. TRANSFERENCIA CRUZADA---")
    cruz = {}
    for orig, alvo in (("serra", "planicie"), ("planicie", "serra")):
        m, mu, sd = modelos[orig]
        s = serra if alvo == "serra" else plano
        Xz = (s[FEATURES].to_numpy(dtype=float) - mu) / sd
        a = float(roc_auc_score(s["classe"].to_numpy().astype(int),
                                m.predict_proba(Xz)[:, 1]))
        proprio = resultados[alvo]["auc_cv"]
        cruz[f"{orig}->{alvo}"] = {"auc": round(a, 4), "auc_nativo_do_alvo": proprio,
                                   "variacao": round(a - proprio, 4)}
        print(f"  {orig:9s} -> {alvo:9s} AUC={a:.4f} "
              f"(nativo do alvo {proprio:.4f}, variacao {a-proprio:+.4f})")

    # ---------------- veredito contra criterios declarados ----------------
    print("---VEREDITO CONTRA CRITERIOS DECLARADOS---")
    falhas = []
    r = resultados["serra"]
    a = r["auc_cv"]
    if a >= AUC_SUSPEITA:
        falhas.append(f"AUC {a:.4f} >= {AUC_SUSPEITA} -- suspeita de vazamento")
    elif not (AUC_MIN <= a <= AUC_MAX):
        falhas.append(f"AUC {a:.4f} fora da faixa esperada [{AUC_MIN}, {AUC_MAX}]")
    if abs(r["gap"]) > GAP_MAX:
        falhas.append(f"gap {r['gap']:+.4f} acima de {GAP_MAX}")
    if r["auc_max_fold"] is not None and r["auc_max_fold"] >= 0.999:
        falhas.append(f"fold com AUC {r['auc_max_fold']:.4f} -- particao degenerada")
    for f, exigido in SINAL_EXIGIDO.items():
        c = r["coef"][f]
        if np.sign(c) != exigido:
            falhas.append(f"{f} com sinal {'+' if c > 0 else '-'} "
                          f"(exigido {'+' if exigido > 0 else '-'})")
        lo, hi = r["ic95"][f]
        if not np.isnan(lo) and lo * hi <= 0:
            falhas.append(f"{f} com IC95 cruzando zero")

    for f in falhas:
        print(f"  REPROVA: {f}")
    veredito = "COERENTE_COM_CRITERIOS" if not falhas else "FORA_DOS_CRITERIOS"
    print(f"VEREDITO={veredito}")

    saida = {"features": FEATURES, "criterios": {
        "auc_faixa": [AUC_MIN, AUC_MAX], "auc_suspeita": AUC_SUSPEITA,
        "gap_max": GAP_MAX, "sinal_exigido": SINAL_EXIGIDO,
        "epv_minimo": EPV_MINIMO},
        "caracterizacao": carac, "modelos": resultados,
        "transferencia": cruz, "falhas": falhas, "veredito": veredito,
        "aois_ingremes": sorted(ingremes), "semente": SEMENTE,
        "n_boot": N_BOOT, "segundos": round(time.time() - t0, 1)}
    (OUT / "resultado.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([
        {"subconjunto": k, "feature": f, "coef": v["coef"][f],
         "ic95_lo": v["ic95"][f][0], "ic95_hi": v["ic95"][f][1],
         "auc_cv": v["auc_cv"], "grupos": v["grupos"], "n": v["n"]}
        for k, v in resultados.items() for f in FEATURES
    ]).to_csv(OUT / "coeficientes.csv", index=False)

    print(f"GRAVADO={OUT}")
    print(f"TEMPO={time.time()-t0:.1f}s")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
