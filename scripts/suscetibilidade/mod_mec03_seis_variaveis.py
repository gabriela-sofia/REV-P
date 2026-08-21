"""
MOD-MEC-03 -- O pool fluvial rodando com as SEIS variaveis, pela primeira vez.

O QUE MUDOU PARA ISTO SER POSSIVEL:

ate agora o modelo multirregiao rodava com quatro variaveis de terreno, e nao
por escolha metodologica: a chuva simplesmente nao existia fora do piloto
ingles e das duas regioes brasileiras. A causa nao era falta de produto de
precipitacao -- ERA5-Land e global -- e sim que `rain_max_24h` se define numa
janela ANTES do evento, e CEMS, Sen1Floods11 e UFO tinham entrado no `ds01`
com `data_evento` fixado em NaT.

O `chuva03` mostrou que as datas nao estavam perdidas, estavam descartadas, e
recuperou 107.558 delas de artefato local. O `chuva04` trouxe a chuva para
todas, na mesma fonte das que ja tinham. Hoje o pool tem 64.989 pontos com as
seis variaveis, uma cadeia de terreno e uma fonte de chuva.

A PERGUNTA QUE ISTO RESPONDE, e que nao dava para fazer antes:

a chuva acrescenta alguma coisa ao que o terreno ja diz? As duas familias
descrevem coisas diferentes -- terreno e condicao permanente, chuva e gatilho
-- e a hipotese do projeto sempre foi que suscetibilidade e a primeira. Se a
chuva nao acrescentar, isso reforca a hipotese em vez de enfraquece-la, e
precisa ser dito assim.

DESENHO: os dois conjuntos correm lado a lado, mesmo split, mesma semente.
    TERRENO   as quatro. E o mod_mec02, repetido aqui para servir de linha de
              base exata sob o mesmo sorteio.
    COMPLETO  as seis.

A diferenca entre os dois AUC e o que a chuva acrescenta. Rodar em execucoes
separadas nao permitiria essa leitura, porque parte da diferenca seria sorteio.

HIERARQUIA DO NEGATIVO aplicada: negativo por ausencia fica FORA, conforme o
`ds01`. O `mod_mec02` ja mostrou que incluir ou nao muda pouco (0,7436 contra
0,7318), entao aqui roda so a versao que a regra autoriza.

NAO faz: nao ajusta hiperparametro, nao seleciona feature por desempenho, nao
mistura cadeia nem fonte de chuva.

Uso:
    python scripts/suscetibilidade/mod_mec03_seis_variaveis.py
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
from ds03_esquema_alvo import VERSAO  # noqa: E402
from mod_serra01_ingreme_2features import (  # noqa: E402
    AUC_MAX, AUC_MIN, AUC_SUSPEITA, EPV_MINIMO, GAP_MAX, N_BOOT, SEMENTE,
    ajustar, padronizar,
)

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ds-05-tabela-unica" / f"tabela_unica_pool_fluvial_{VERSAO}.csv"
OUT = RUNS / "mod-mec-03"

TERRENO = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
CHUVA = ["rain_max_24h", "rain_decay_index"]
CONJUNTOS = {"TERRENO": TERRENO, "COMPLETO": TERRENO + CHUVA}

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

    POR QUE O IC NAO E ENFEITE (Ploton et al., 2020, Nat. Commun. 11:4540):
    sem intervalo, 0,7382 e 0,6100 parecem igualmente solidos, e nao sao. Numa
    matriz de decisao onde o limiar e 0,60, uma aprovacao que encosta no limiar
    e uma que sobra precisam ser distinguiveis -- caso contrario o limiar
    decide sozinho, sem dizer com quanta confianca.

    O bootstrap reamostra GRUPO do conjunto de teste, nao linha. Ploton mostra
    que com dado espacialmente autocorrelacionado tratar pontos vizinhos como
    independentes infla a confianca; o mecanismo que estreita o IC e o mesmo
    que infla a validacao cruzada aleatoria. Aqui o grupo e evento, AOI ou
    chip, entao reamostra-lo preserva o bloco espacial.

    O modelo fica FIXO durante o bootstrap. O que se mede e a variabilidade do
    ESTIMADOR sobre a amostra de teste, e nao a instabilidade do ajuste -- essa
    ja esta medida no IC dos coeficientes.
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
    print("===MOD-MEC-03: POOL FLUVIAL COM AS SEIS VARIAVEIS===")

    if not ENTRADA.exists():
        print(f"ABORTADO: {ENTRADA} ausente. Rode ds05.")
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
                                 grupos=("grupo_cv", "nunique")).to_string())

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

    # ---- o que a chuva acrescenta ----
    delta = res["COMPLETO"]["auc_cv"] - res["TERRENO"]["auc_cv"]
    print(f"\n{'='*68}\n--- O QUE A CHUVA ACRESCENTA ---")
    print(f"  so terreno: {res['TERRENO']['auc_cv']:.4f}")
    print(f"  com chuva:  {res['COMPLETO']['auc_cv']:.4f}")
    print(f"  ganho = {delta:+.4f}  (mesmo split, mesma semente)")
    chuva_ic = {f: res["COMPLETO"]["ic95"][f] for f in CHUVA}
    cruza = [f for f, (lo, hi) in chuva_ic.items() if lo * hi <= 0]
    if abs(delta) < 0.01 and len(cruza) == len(CHUVA):
        leitura = ("a chuva nao acrescenta discriminacao ao terreno neste pool, "
                   "e nenhum dos dois coeficientes se distingue de zero. Isso e "
                   "coerente com a hipotese do projeto: suscetibilidade e "
                   "condicao permanente, e o gatilho nao a explica")
    elif delta >= 0.01:
        leitura = (f"a chuva acrescenta {delta:.4f} de AUC. O gatilho carrega "
                   "informacao que o terreno nao tem")
    else:
        leitura = ("a chuva nao melhora o AUC agregado, mas ao menos um "
                   f"coeficiente se distingue de zero ({[f for f in CHUVA if f not in cruza]})")
    print(f"  LEITURA: {leitura}")

    # ---- transferencia, com o conjunto completo ----
    feats = CONJUNTOS["COMPLETO"]
    print("\n--- LEAVE-ONE-SOURCE-OUT (seis variaveis) ---")
    loso = transferir(d, feats, "fonte")
    for x in loso:
        print(f"  sem {x['estrato']:14s} -> "
              + (f"AUC={x['auc']:.4f} IC95={x.get('ic95')} "
                 f"(n={x['n_teste']:,}, grupos={x.get('grupos_teste')})"
                 if x["auc"] else f"INDEFINIDO ({x.get('motivo')})"))
    print("\n--- TRANSFERENCIA ENTRE CLASSES DE RELEVO (seis variaveis) ---")
    relevo = transferir(d, feats, "classe_relevo")
    for x in relevo:
        print(f"  treinar fora de {x['estrato']:20s} -> "
              + (f"AUC={x['auc']:.4f} IC95={x.get('ic95')} (n={x['n_teste']:,})"
                 if x["auc"] else "INDEFINIDO"))

    saida = {
        "entrada": str(ENTRADA.relative_to(REPO)),
        "n": int(len(d)), "grupos": int(d.grupo_cv.nunique()),
        "fontes": sorted(d.fonte.unique()),
        "fonte_chuva": d.fonte_chuva.unique().tolist(),
        "cadeia_terreno": d.cadeia_terreno.unique().tolist(),
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
