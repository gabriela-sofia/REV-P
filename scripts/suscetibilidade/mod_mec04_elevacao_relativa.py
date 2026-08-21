"""
MOD-MEC-04 -- O pool fluvial com elevacao RELATIVA, sucessor permanente do mod_mec03.

POR QUE ESTE SCRIPT EXISTE, E O QUE ELE NAO INVALIDA:

`app01-transferencia-curitiba-sem-rotulo-local` (19/08/2026) achou que
`elevation_m` absoluta nao e comparavel entre fontes de altitude de base
diferente: Curitiba (~900 m) tinha 0% dos pontos dentro do intervalo 5-95%
de um treino externo perto do nivel do mar (diferenca padronizada de media
= 2,76). O `ds03` (v2) e o `ds04` ja carregam a correcao permanente --
`elevation_rel_m = elevation_m - elevation_baseline_m`, baseline = P1 de
elevation_m dentro da propria fonte, calculada para toda fonte por
construcao (`moldar()`, sem excecao).

Este script troca `elevation_m` por `elevation_rel_m` no conjunto TERRENO.
Ele NAO reabre nem invalida o `mod_mec03/resultado.json` ja publicado: aquele
resultado usou elevacao absoluta, dentro de fontes que -- MEDIDO DEPOIS --
tinham 90%+ do dominio se sobrepondo mesmo sem a correcao (so Curitiba, que
nao entra no pool fluvial multirregiao do mod_mec03 como fonte de TREINO
isolada nesse sentido, tinha o problema extremo). O TERRENO-vs-COMPLETO do
mod_mec03 fica como registro historico de uma pergunta especifica (o que a
chuva acrescenta), respondida sob a variavel que existia na hora. Este script
responde a uma pergunta diferente: a partir de agora, qual e o conjunto
CORRETO para qualquer ajuste que agrupe mais de uma fonte -- e a resposta
permanente e `VARIAVEIS_TERRENO_TRANSFERIVEL`, nao `VARIAVEIS_TERRENO`.

MESMO DESENHO do mod_mec03 (GroupKFold, bootstrap por grupo, LOSO, transferencia
por relevo) -- so a coluna de elevacao muda. Isso e deliberado: qualquer
diferenca de numero entre os dois scripts e atribuivel so a troca de variavel,
nao a mudanca de metodo.

NAO faz: nao ajusta hiperparametro, nao seleciona feature por desempenho, nao
mistura cadeia nem fonte de chuva.

Uso:
    python scripts/suscetibilidade/mod_mec04_elevacao_relativa.py
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
from ds03_esquema_alvo import VARIAVEIS_TERRENO_TRANSFERIVEL, VERSAO  # noqa: E402
from mod_serra01_ingreme_2features import (  # noqa: E402
    AUC_MAX, AUC_MIN, AUC_SUSPEITA, EPV_MINIMO, GAP_MAX, N_BOOT, SEMENTE,
    ajustar, padronizar,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ds-05-tabela-unica" / f"tabela_unica_pool_fluvial_{VERSAO}.csv"
OUT = RUNS / "mod-mec-04"

TERRENO = list(VARIAVEIS_TERRENO_TRANSFERIVEL)  # elevation_rel_m no lugar de elevation_m
CHUVA = ["rain_max_24h", "rain_decay_index"]
CONJUNTOS = {"TERRENO": TERRENO, "COMPLETO": TERRENO + CHUVA}

# mesma exigencia de sinal do mod_mec03 -- elevacao (absoluta ou relativa)
# nunca teve sinal fisico exigido ali, e continua sem ter aqui: relativa muda
# a comparabilidade entre fontes, nao o que se espera do coeficiente.
SINAL_EXIGIDO = {"hand_m": -1, "twi_dinf": +1}
NEGATIVO_ADMITIDO = ("observado", "exclusao_qualificada", "nao_aplicavel")
N_FOLDS = 5


def avaliar(d: pd.DataFrame, feats: list[str], folds: list, rng) -> dict:
    from sklearn.metrics import roc_auc_score

    y = d.classe.to_numpy().astype(int)
    X = d[feats].to_numpy(dtype=float)
    Xz, _, _ = padronizar(X)

    aucs, aucs_tr = [], []
    for tr, te in folds:
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        m = ajustar(Xz[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(Xz[te])[:, 1]))
        aucs_tr.append(roc_auc_score(y[tr], m.predict_proba(Xz[tr])[:, 1]))
    auc = float(np.mean(aucs))
    gap = float(np.mean(aucs_tr) - auc)

    coef = ajustar(Xz, y).coef_.ravel()[:len(feats)]
    grupos = d.grupo_cv.unique()
    idx = {g: np.flatnonzero((d.grupo_cv == g).to_numpy()) for g in grupos}
    boots = []
    for _ in range(N_BOOT):
        esc = rng.choice(grupos, size=len(grupos), replace=True)
        linhas = np.concatenate([idx[g] for g in esc])
        ys = y[linhas]
        if len(np.unique(ys)) < 2:
            continue
        try:
            Xs, _, _ = padronizar(X[linhas])
            boots.append(ajustar(Xs, ys).coef_.ravel()[:len(feats)])
        except Exception:  # noqa: BLE001
            continue
    boots = np.array(boots)
    ic = (np.percentile(boots, [2.5, 97.5], axis=0) if len(boots) > 30
          else np.full((2, len(feats)), np.nan))

    falhas = []
    if auc >= AUC_SUSPEITA:
        falhas.append(f"AUC {auc:.4f} >= {AUC_SUSPEITA} (vazamento)")
    elif not (AUC_MIN <= auc <= AUC_MAX):
        falhas.append(f"AUC {auc:.4f} fora de [{AUC_MIN}, {AUC_MAX}]")
    if abs(gap) > GAP_MAX:
        falhas.append(f"gap {gap:+.4f} acima de {GAP_MAX}")
    for f, exigido in SINAL_EXIGIDO.items():
        i = feats.index(f)
        if np.sign(coef[i]) != exigido:
            falhas.append(f"{f} com sinal invertido")
        if not np.isnan(ic[0, i]) and ic[0, i] * ic[1, i] <= 0:
            falhas.append(f"{f} com IC95 cruzando zero")

    return {"features": feats, "n_features": len(feats),
            "epv": round(d.grupo_cv.nunique() / len(feats), 1),
            "auc_cv": round(auc, 4), "folds": len(aucs),
            "auc_min": round(min(aucs), 4), "auc_max": round(max(aucs), 4),
            "gap": round(gap, 4),
            "coef": {f: round(float(c), 4) for f, c in zip(feats, coef)},
            "ic95": {f: [round(float(ic[0, i]), 4), round(float(ic[1, i]), 4)]
                     for i, f in enumerate(feats)},
            "falhas": falhas,
            "veredito": "COERENTE_COM_CRITERIOS" if not falhas else "FORA_DOS_CRITERIOS"}


def _auc_posto(y: np.ndarray, s: np.ndarray) -> float | None:
    """AUC por posto. No laco de bootstrap isto roda milhares de vezes."""
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def transferir(d: pd.DataFrame, feats: list[str], coluna: str,
               n_boot: int = 2000) -> list[dict]:
    """Treina fora do estrato, testa dentro, com IC no estimador.

    Mesmo desenho e mesma justificativa (Ploton et al. 2020) do mod_mec03 --
    reproduzido aqui, nao reimportado, para que os dois scripts fiquem
    autocontidos e comparaveis linha a linha.
    """
    from sklearn.metrics import roc_auc_score

    y = d.classe.to_numpy().astype(int)
    X = d[feats].to_numpy(dtype=float)
    rng = np.random.default_rng(SEMENTE)
    saida = []
    for valor in sorted(d[coluna].dropna().unique()):
        te = (d[coluna] == valor).to_numpy()
        tr = ~te
        linha = {"estrato": str(valor), "n_teste": int(te.sum()),
                 "pos_teste": int(y[te].sum()),
                 "grupos_teste": int(d.loc[te, "grupo_cv"].nunique())}
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2 or tr.sum() < 50:
            linha["auc"] = None
            linha["ic95"] = None
            linha["motivo"] = "estrato ou treino sem as duas classes"
            saida.append(linha)
            continue

        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd == 0, 1, sd)
        m = ajustar((X[tr] - mu) / sd, y[tr])
        p = m.predict_proba((X[te] - mu) / sd)[:, 1]
        yte = y[te]
        linha["auc"] = round(float(roc_auc_score(yte, p)), 4)

        grupos = d.loc[te, "grupo_cv"].to_numpy()
        unicos = pd.unique(grupos)
        idx = {g: np.flatnonzero(grupos == g) for g in unicos}
        amostras = []
        for _ in range(n_boot):
            esc = rng.choice(unicos, size=len(unicos), replace=True)
            linhas = np.concatenate([idx[g] for g in esc])
            v = _auc_posto(yte[linhas], p[linhas])
            if v is not None:
                amostras.append(v)
        if len(amostras) >= 30:
            lo, hi = np.percentile(amostras, [2.5, 97.5])
            linha["ic95"] = [round(float(lo), 4), round(float(hi), 4)]
            linha["n_grupos_bootstrap"] = int(len(unicos))
        else:
            linha["ic95"] = None
            linha["motivo_sem_ic"] = "menos de 30 reamostragens validas"
        saida.append(linha)
    return saida


def main() -> int:
    from sklearn.model_selection import GroupKFold

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("===MOD-MEC-04: POOL FLUVIAL COM ELEVACAO RELATIVA===")

    if not ENTRADA.exists():
        print(f"ABORTADO: {ENTRADA} ausente. Rode ds04+ds05 (esquema {VERSAO}).")
        return 1
    d = pd.read_csv(ENTRADA, low_memory=False)
    d = d[d.classe.isin([0, 1])
          & d.nivel_negativo.isin(NEGATIVO_ADMITIDO)].copy()
    todas = TERRENO + CHUVA
    antes = len(d)
    d = d.dropna(subset=todas + ["grupo_cv"]).reset_index(drop=True)

    print(f"n={len(d):,} (descartados {antes-len(d):,} sem as seis) "
          f"grupos={d.grupo_cv.nunique():,} fontes={d.fonte.nunique()}")
    print(f"fonte de chuva: {d.fonte_chuva.unique().tolist()}")
    print(f"cadeia de terreno: {d.cadeia_terreno.unique().tolist()}")
    if d.fonte_chuva.nunique() > 1 or d.cadeia_terreno.nunique() > 1:
        print("ABORTADO: mais de uma procedencia no conjunto. Isso invalidaria "
              "a comparacao entre os dois blocos.")
        return 2
    print(d.groupby("fonte").agg(n=("classe", "size"), pos=("classe", "sum"),
                                 grupos=("grupo_cv", "nunique"),
                                 elev_baseline=("elevation_baseline_m", "mean"),
                                 ).to_string())

    epv = d.grupo_cv.nunique() / len(todas)
    if epv < EPV_MINIMO:
        print(f"ABORTADO: EPV {epv:.1f} abaixo de {EPV_MINIMO}")
        return 3

    # MESMO split para os dois conjuntos: a diferenca de AUC tem de ser a
    # chuva, e nao o sorteio das dobras
    y = d.classe.to_numpy().astype(int)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(
        d[todas].to_numpy(dtype=float), y, groups=d.grupo_cv.to_numpy()))

    res = {}
    for nome, feats in CONJUNTOS.items():
        r = avaliar(d, feats, folds, np.random.default_rng(SEMENTE))
        res[nome] = r
        print(f"\n--- {nome} ({r['n_features']} features, EPV={r['epv']}) ---")
        print(f"  AUC_CV={r['auc_cv']:.4f} ({r['auc_min']:.4f}-{r['auc_max']:.4f}) "
              f"gap={r['gap']:+.4f}")
        for f in feats:
            lo, hi = r["ic95"][f]
            marca = "  <-- IC CRUZA ZERO" if lo * hi <= 0 else ""
            print(f"    {f:20s} coef={r['coef'][f]:+8.4f} "
                  f"IC95=[{lo:+7.4f},{hi:+7.4f}]{marca}")
        for x in r["falhas"]:
            print(f"  REPROVA: {x}")
        print(f"  VEREDITO={r['veredito']}")

    delta = res["COMPLETO"]["auc_cv"] - res["TERRENO"]["auc_cv"]
    print(f"\n{'='*68}\n--- O QUE A CHUVA ACRESCENTA (sob elevacao relativa) ---")
    print(f"  so terreno: {res['TERRENO']['auc_cv']:.4f}")
    print(f"  com chuva:  {res['COMPLETO']['auc_cv']:.4f}")
    print(f"  ganho = {delta:+.4f}  (mesmo split, mesma semente)")
    chuva_ic = {f: res["COMPLETO"]["ic95"][f] for f in CHUVA}
    cruza = [f for f, (lo, hi) in chuva_ic.items() if lo * hi <= 0]
    if abs(delta) < 0.01 and len(cruza) == len(CHUVA):
        leitura = ("a chuva nao acrescenta discriminacao ao terreno neste pool, "
                   "e nenhum dos dois coeficientes se distingue de zero")
    elif delta >= 0.01:
        leitura = (f"a chuva acrescenta {delta:.4f} de AUC")
    else:
        leitura = ("a chuva nao melhora o AUC agregado, mas ao menos um "
                   f"coeficiente se distingue de zero ({[f for f in CHUVA if f not in cruza]})")
    print(f"  LEITURA: {leitura}")

    feats = CONJUNTOS["COMPLETO"]
    print("\n--- LEAVE-ONE-SOURCE-OUT (elevacao relativa) ---")
    loso = transferir(d, feats, "fonte")
    for x in loso:
        print(f"  sem {x['estrato']:14s} -> "
              + (f"AUC={x['auc']:.4f} IC95={x.get('ic95')} "
                 f"(n={x['n_teste']:,}, grupos={x.get('grupos_teste')})"
                 if x["auc"] else f"INDEFINIDO ({x.get('motivo')})"))
    print("\n--- TRANSFERENCIA ENTRE CLASSES DE RELEVO (elevacao relativa) ---")
    relevo = transferir(d, feats, "classe_relevo")
    for x in relevo:
        print(f"  treinar fora de {x['estrato']:20s} -> "
              + (f"AUC={x['auc']:.4f} IC95={x.get('ic95')} (n={x['n_teste']:,})"
                 if x["auc"] else "INDEFINIDO"))

    saida = {
        "entrada": str(ENTRADA.relative_to(REPO)),
        "sucede": "mod-mec-03 (elevation_m absoluta) -- ver docstring deste arquivo",
        "n": int(len(d)), "grupos": int(d.grupo_cv.nunique()),
        "fontes": sorted(d.fonte.unique()),
        "fonte_chuva": d.fonte_chuva.unique().tolist(),
        "cadeia_terreno": d.cadeia_terreno.unique().tolist(),
        "variavel_elevacao": "elevation_rel_m (relativa ao P1 da propria fonte)",
        "conjuntos": res,
        "ganho_da_chuva": round(delta, 4), "leitura_da_chuva": leitura,
        "leave_one_source_out": loso,
        "transferencia_relevo": relevo,
        "semente": SEMENTE, "segundos": round(time.time() - t0, 1),
        "nao_e": ("nao e validacao operacional nem autoriza uso preditivo: e "
                  "um ajuste com validacao agrupada sobre a tabela unica"),
    }
    (OUT / "resultado.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
