"""
MOD-SERRA-02 -- O mesmo modelo, os mesmos pontos, dois instrumentos.

A PERGUNTA:
o `ter02` mostrou que as duas cadeias de derivacao concordam de forma muito
desigual entre features, medido no nivel do ponto amostrado:

    elevation_m   pearson 0,999    praticamente identico
    hand_m        pearson 0,935    concordancia alta
    slope_deg     pearson 0,905    concordancia alta
    twi_dinf      pearson 0,633    DISCORDANCIA GRANDE

TWI e a feature menos reproduzivel entre implementacoes -- e e justamente a
que mais pesou no modelo de serra (coeficiente +0,50, tres vezes o da
planicie). Se a conclusao sobre encosta depender de TWI, e TWI depende do
software, a conclusao depende do software.

Este script isola essa duvida do jeito mais limpo possivel: **mesmos pontos,
mesmos grupos, mesma especificacao, mesma semente -- so muda o instrumento.**
Qualquer diferenca de coeficiente ou de AUC e atribuivel a cadeia de
derivacao, e a nada mais.

O QUE UM RESULTADO OU OUTRO SIGNIFICA:

  coeficientes proximos  ->  a conclusao do mod-serra-01 e robusta a
                             instrumento; o TWI discorda em nivel mas
                             preserva a ordenacao que o modelo usa.
  coeficientes distantes ->  parte do que foi lido como fisica de encosta era
                             artefato de implementacao. Nesse caso vale a
                             cadeia WBT, porque e a que Petropolis usa -- e a
                             comparabilidade com o alvo e o criterio, nao o AUC.

CRITERIO DE DESEMPATE, declarado antes: **nao se escolhe a cadeia pelo AUC.**
Escolhe-se a que e comparavel com Petropolis, que ja tem derivacao
WhiteboxTools. Um AUC maior na cadeia global seria irrelevante, porque o
modelo nao poderia ser aplicado ao alvo sem trocar de variavel no meio.

NAO faz: nao ajusta hiperparametro, nao seleciona feature, nao declara modelo
final.

Uso:
    python scripts/suscetibilidade/mod_serra02_instrumento.py
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
    FEATURES, EPV_MINIMO, N_BOOT, SEMENTE, AUC_MIN, AUC_MAX, AUC_SUSPEITA,
    GAP_MAX, SINAL_EXIGIDO, ajustar, padronizar, avaliar_cv, bootstrap_por_grupo,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ter-02-comparacao" / "dataset_serra_harmonizado.csv"
OUT = RUNS / "mod-serra-02"


def rodar(sub: pd.DataFrame, rotulo: str, rng) -> dict:
    val, tr = avaliar_cv(sub)
    Xz, _, _ = padronizar(sub[FEATURES].to_numpy(dtype=float))
    m = ajustar(Xz, sub["classe"].to_numpy().astype(int))
    coef = m.coef_.ravel()[:len(FEATURES)]
    boot = bootstrap_por_grupo(sub, rng)
    ic = (np.percentile(boot, [2.5, 97.5], axis=0)
          if len(boot) > 30 else np.full((2, len(FEATURES)), np.nan))
    auc = float(np.mean(val)) if len(val) else float("nan")
    gap = float(np.mean(tr) - np.mean(val)) if len(val) else float("nan")

    print(f"  {rotulo:10s} AUC_CV={auc:.4f} (folds={len(val)}) gap={gap:+.4f}")
    for i, f in enumerate(FEATURES):
        lo, hi = ic[0, i], ic[1, i]
        marca = "" if (np.isnan(lo) or lo * hi > 0) else "  <-- IC CRUZA ZERO"
        print(f"      {f:10s} coef={coef[i]:+8.4f}  IC95=[{lo:+7.4f},{hi:+7.4f}]{marca}")

    return {"auc_cv": round(auc, 4), "gap": round(gap, 4), "folds": len(val),
            "auc_max_fold": round(float(val.max()), 4) if len(val) else None,
            "n": int(len(sub)), "grupos": int(sub.grupo_cv.nunique()),
            "coef": {f: round(float(c), 4) for f, c in zip(FEATURES, coef)},
            "ic95": {f: [round(float(ic[0, i]), 4), round(float(ic[1, i]), 4)]
                     for i, f in enumerate(FEATURES)}}


def veredito(r: dict) -> list[str]:
    falhas = []
    a = r["auc_cv"]
    if a >= AUC_SUSPEITA:
        falhas.append(f"AUC {a:.4f} >= {AUC_SUSPEITA} (vazamento)")
    elif not (AUC_MIN <= a <= AUC_MAX):
        falhas.append(f"AUC {a:.4f} fora de [{AUC_MIN}, {AUC_MAX}]")
    if abs(r["gap"]) > GAP_MAX:
        falhas.append(f"gap {r['gap']:+.4f} acima de {GAP_MAX}")
    if r["auc_max_fold"] is not None and r["auc_max_fold"] >= 0.999:
        falhas.append(f"fold com AUC {r['auc_max_fold']:.4f}")
    for f, exigido in SINAL_EXIGIDO.items():
        c = r["coef"][f]
        if np.sign(c) != exigido:
            falhas.append(f"{f} com sinal invertido")
        lo, hi = r["ic95"][f]
        if not np.isnan(lo) and lo * hi <= 0:
            falhas.append(f"{f} com IC95 cruzando zero")
    return falhas


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("===MOD-SERRA-02: MESMOS PONTOS, DOIS INSTRUMENTOS===")

    if not ENTRADA.exists():
        print(f"ABORTADO: {ENTRADA} ausente. Rode ter02 antes.")
        return 1
    df = pd.read_csv(ENTRADA, low_memory=False)

    cols_wbt = [f"{c}_wbt" for c in FEATURES]
    cols_glo = [f"{c}_global" for c in FEATURES]
    faltam = [c for c in cols_wbt + cols_glo if c not in df.columns]
    if faltam:
        print(f"ABORTADO: colunas ausentes: {faltam}")
        return 1

    df = df.dropna(subset=cols_wbt + cols_glo + ["classe", "grupo_cv"])
    df = df[df.classe.isin([0, 1])].reset_index(drop=True)
    epv = df.grupo_cv.nunique() / len(FEATURES)
    print(f"PONTOS={len(df):,} pos={int(df.classe.sum()):,} "
          f"neg={len(df)-int(df.classe.sum()):,} grupos={df.grupo_cv.nunique()} "
          f"EPV={epv:.1f}")
    if epv < EPV_MINIMO:
        print(f"ABORTADO: EPV {epv:.1f} abaixo de {EPV_MINIMO}.")
        return 3

    print("---MODELOS---")
    res = {}
    for rotulo, cols in (("global", cols_glo), ("wbt", cols_wbt)):
        sub = df.copy()
        for f, c in zip(FEATURES, cols):
            sub[f] = sub[c]
        res[rotulo] = rodar(sub, rotulo, np.random.default_rng(SEMENTE))

    print("---DIFERENCA ATRIBUIVEL AO INSTRUMENTO---")
    d_auc = res["wbt"]["auc_cv"] - res["global"]["auc_cv"]
    print(f"  delta AUC (wbt - global) = {d_auc:+.4f}")
    concorda = True
    for f in FEATURES:
        cg, cw = res["global"]["coef"][f], res["wbt"]["coef"][f]
        lg, hg = res["global"]["ic95"][f]
        lw, hw = res["wbt"]["ic95"][f]
        sobrepoe = not (hw < lg or hg < lw)
        concorda &= sobrepoe and np.sign(cg) == np.sign(cw)
        print(f"  {f:10s} global={cg:+7.4f}  wbt={cw:+7.4f}  "
              f"delta={cw-cg:+7.4f}  IC_sobrepoe={'sim' if sobrepoe else 'NAO'}")

    print("---VEREDITO---")
    for rotulo in ("global", "wbt"):
        f = veredito(res[rotulo])
        print(f"  {rotulo:8s} {'COERENTE_COM_CRITERIOS' if not f else 'FORA: ' + '; '.join(f)}")
    print(f"  robusto_ao_instrumento={'SIM' if concorda else 'NAO'}")
    print("  cadeia adotada = WBT (comparabilidade com Petropolis, nao AUC)")

    (OUT / "resultado.json").write_text(json.dumps({
        "features": FEATURES, "n": int(len(df)),
        "grupos": int(df.grupo_cv.nunique()), "epv": round(epv, 2),
        "modelos": res, "delta_auc_wbt_menos_global": round(d_auc, 4),
        "robusto_ao_instrumento": bool(concorda),
        "cadeia_adotada": "wbt",
        "motivo_adocao": ("Petropolis ja tem derivacao WhiteboxTools; treinar no "
                          "produto global e predizer no WBT trocaria a variavel "
                          "no meio. Criterio e comparabilidade, nao desempenho."),
        "veredito_global": veredito(res["global"]),
        "veredito_wbt": veredito(res["wbt"]),
        "semente": SEMENTE, "n_boot": N_BOOT,
        "segundos": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
