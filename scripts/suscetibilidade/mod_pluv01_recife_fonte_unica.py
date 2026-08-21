"""
MOD-PLUV-01 -- O modelo pluvial de Recife refeito sobre chuva de fonte unica.

POR QUE ISTO PRECISA EXISTIR, e por que nao e opcional:

o registro de regioes do MVP anuncia Recife como a unica regiao `available`,
com "SUSC-20 v12 (Firth, n=269, LOO-AUC=0.6781)". Esse numero foi calculado
sobre uma coluna de chuva que carregava DUAS fontes -- 181 pontos em CHIRPS
v2.0 e 97 em Open-Meteo/ERA5-Land -- e o `aud_chuva01` mediu que a fonte
estava associada ao rotulo: 80% de positivos no estrato CHIRPS contra 9% no
estrato ERA5, com o indicador de proveniencia discriminando 0,8256 contra
0,7377 da propria chuva.

Como a chuva antecedente e o preditor principal da trilha pluvial, parte do
0,6781 podia ser proveniencia e nao precipitacao. Nao dava para saber quanto
sem refazer. O `chuva02` reamostrou os 181 pontos para a fonte unica; este
script e a consequencia: mede o mesmo modelo, no mesmo desenho, com a coluna
agora limpa.

O RESULTADO E O QUE FOR. Se o AUC cair, o MVP estava anunciando um numero que
media parcialmente a campanha de aquisicao, e o registro precisa ser corrigido
-- que e exatamente o motivo de rodar isto antes de propor qualquer regiao
nova para a posicao de MVP.

DESENHO, deliberadamente identico ao do v12 para que a comparacao signifique
alguma coisa: Firth, leave-one-out, mesmas features. A unica coisa que muda e
a procedencia da chuva. Mudar mais de uma coisa por vez tornaria a diferenca
ininterpretavel.

TRES CONJUNTOS DE FEATURES, reportados lado a lado, porque a pergunta "o
modelo ainda funciona?" se decompoe em tres:
    SO_CHUVA      as duas de precipitacao. Isola o preditor principal.
    SO_TERRENO    as quatro de terreno. E o controle: em Recife o terreno
                  declaradamente nao e o mecanismo (HAND com p = 0,978), entao
                  este bloco deve ficar perto do acaso -- se nao ficar, a
                  premissa da separacao por mecanismo e que esta errada.
    COMPLETO      as seis. E o que o projeto passa a poder rodar em toda
                  regiao depois do chuva04.

NAO faz: nao ajusta hiperparametro, nao seleciona feature por desempenho, nao
altera o v12 e nao escreve no registro de regioes -- corrigir o registro e
decisao da Gabriela, com o numero na mao.

Uso:
    python scripts/suscetibilidade/mod_pluv01_recife_fonte_unica.py
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
from mod_serra01_ingreme_2features import N_BOOT, SEMENTE, ajustar, padronizar  # noqa: E402

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
RUNS = REPO / "local_runs"
ENTRADA = RUNS / "ds-04-reducao" / "recife.csv"
OUT = RUNS / "mod-pluv-01"

TERRENO = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
CHUVA = ["rain_max_24h", "rain_decay_index"]

CONJUNTOS = {
    "SO_CHUVA": CHUVA,
    "SO_TERRENO": TERRENO,
    "COMPLETO": TERRENO + CHUVA,
}

# numero publicado no registro de regioes, sobre a chuva de fonte misturada
V12_LOO_AUC = 0.6781
V12_N = 269


def loo_auc(X: np.ndarray, y: np.ndarray) -> float | None:
    """Leave-one-out, igual ao v12. Padroniza DENTRO de cada dobra.

    Padronizar antes do split usaria a media e o desvio do ponto que esta
    sendo previsto -- vazamento pequeno e real, e num n de 269 ele aparece.
    """
    from sklearn.metrics import roc_auc_score

    pred = np.full(len(y), np.nan)
    for i in range(len(y)):
        tr = np.ones(len(y), dtype=bool)
        tr[i] = False
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd == 0, 1, sd)
        try:
            m = ajustar((X[tr] - mu) / sd, y[tr])
            pred[i] = m.predict_proba(((X[i] - mu) / sd).reshape(1, -1))[0, 1]
        except Exception:  # noqa: BLE001
            continue
    ok = np.isfinite(pred)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], pred[ok]))


def ic_bootstrap(X: np.ndarray, y: np.ndarray, rng) -> list[float] | None:
    """IC do coeficiente por reamostragem de pontos (o grupo aqui e o ponto)."""
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            Xs, _, _ = padronizar(X[i])
            boots.append(ajustar(Xs, y[i]).coef_.ravel()[:X.shape[1]])
        except Exception:  # noqa: BLE001
            continue
    return np.array(boots) if len(boots) > 30 else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)
    t0 = time.time()
    print("===MOD-PLUV-01: RECIFE COM CHUVA DE FONTE UNICA===")

    if not ENTRADA.exists():
        print(f"ABORTADO: {ENTRADA} ausente. Rode ds04.")
        return 1
    d = pd.read_csv(ENTRADA, low_memory=False)
    d = d[d.classe.isin([0, 1])].copy()

    print(f"fonte de chuva: {d.fonte_chuva.value_counts().to_dict()}")
    if d.loc[d.rain_max_24h.notna(), "fonte_chuva"].nunique() > 1:
        print("ABORTADO: ainda ha mais de uma fonte de chuva com valor. "
              "Rode chuva02 antes -- este script existe para medir o caso "
              "de fonte unica.")
        return 2

    completo = d[TERRENO + CHUVA].notna().all(axis=1)
    print(f"n total={len(d)}  com as 6 variaveis={int(completo.sum())}  "
          f"(descartados {int((~completo).sum())} sem chuva na origem)")
    d = d[completo].reset_index(drop=True)
    y = d.classe.to_numpy().astype(int)
    print(f"positivos={int(y.sum())} negativos={int(len(y)-y.sum())}")
    print(f"cadeia de terreno: {d.cadeia_terreno.unique().tolist()}")

    resultados = {}
    for nome, feats in CONJUNTOS.items():
        X = d[feats].to_numpy(dtype=float)
        epv = min(int(y.sum()), int(len(y) - y.sum())) / len(feats)
        auc = loo_auc(X, y)
        Xz, _, _ = padronizar(X)
        coef = ajustar(Xz, y).coef_.ravel()[:len(feats)]
        boots = ic_bootstrap(X, y, rng)
        ic = (np.percentile(boots, [2.5, 97.5], axis=0) if boots is not None
              else np.full((2, len(feats)), np.nan))

        print(f"\n--- {nome} ({len(feats)} features, EPV={epv:.1f}) ---")
        print(f"  LOO-AUC = {auc:.4f}" if auc is not None else "  LOO-AUC indefinido")
        for i, f in enumerate(feats):
            lo, hi = ic[0, i], ic[1, i]
            marca = "" if (np.isnan(lo) or lo * hi > 0) else "  <-- IC CRUZA ZERO"
            print(f"    {f:20s} coef={coef[i]:+8.4f} "
                  f"IC95=[{lo:+7.4f},{hi:+7.4f}]{marca}")
        resultados[nome] = {
            "features": feats, "n": int(len(y)), "epv": round(epv, 1),
            "loo_auc": round(auc, 4) if auc is not None else None,
            "coef": {f: round(float(c), 4) for f, c in zip(feats, coef)},
            "ic95": {f: [round(float(ic[0, i]), 4), round(float(ic[1, i]), 4)]
                     for i, f in enumerate(feats)},
        }

    # ---- a comparacao que motivou o script ----
    print(f"\n{'='*68}\n--- O QUE MUDOU EM RELACAO AO NUMERO PUBLICADO ---")
    novo = resultados["COMPLETO"]["loo_auc"]
    print(f"  v12, chuva de fonte MISTURADA (n={V12_N}): LOO-AUC = {V12_LOO_AUC:.4f}")
    print(f"  agora, chuva de fonte UNICA   (n={resultados['COMPLETO']['n']}): "
          f"LOO-AUC = {novo:.4f}" if novo else "  agora: indefinido")
    if novo is not None:
        delta = novo - V12_LOO_AUC
        print(f"  diferenca = {delta:+.4f}")
        so_chuva = resultados["SO_CHUVA"]["loo_auc"]
        so_terr = resultados["SO_TERRENO"]["loo_auc"]
        if abs(delta) < 0.03:
            leitura = ("o numero publicado se sustenta com a fonte limpa: a "
                       "discriminacao nao vinha da proveniencia")
        elif delta < 0:
            leitura = (f"o numero publicado CAI {abs(delta):.4f} com a fonte "
                       "limpa. Parte do que o v12 media era a campanha de "
                       "aquisicao, nao precipitacao. O registro de regioes "
                       "precisa ser corrigido antes de anunciar Recife")
        else:
            leitura = (f"o numero SOBE {delta:.4f} com a fonte limpa -- a "
                       "mistura estava atrapalhando, nao ajudando")
        print(f"  LEITURA: {leitura}")
        print(f"\n  decomposicao: so chuva={so_chuva:.4f}  "
              f"so terreno={so_terr:.4f}  completo={novo:.4f}")
        if so_terr is not None and abs(so_terr - 0.5) < 0.08:
            print("  o bloco de terreno fica perto do acaso, como a separacao "
                  "por mecanismo previa: em Recife o terreno nao e o mecanismo")
        resultados["comparacao"] = {
            "v12_loo_auc": V12_LOO_AUC, "v12_n": V12_N,
            "agora_loo_auc": novo, "delta": round(delta, 4), "leitura": leitura}

    resultados["fonte_chuva"] = d.fonte_chuva.unique().tolist()
    resultados["esquema"] = VERSAO
    resultados["semente"] = SEMENTE
    resultados["segundos"] = round(time.time() - t0, 1)
    resultados["nao_e"] = (
        "nao e validacao operacional. E o mesmo desenho do v12 com a coluna de "
        "chuva corrigida, para medir quanto da discriminacao publicada era "
        "proveniencia")
    (OUT / "resultado.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
