"""
MVP-01 -- Que regioes podem ocupar a posicao de MVP, medido e nao afirmado.

A PERGUNTA:

o registro de regioes tem Recife como unica `available` e Curitiba como
`limited_evidence`. Esse registro foi escrito antes da cadeia harmonizada, do
pool multirregiao e da correcao de chuva. Vale perguntar de novo: alguma outra
regiao ja da para ocupar a posicao, e se nao, seria util testar alguma la?

O QUE DECIDE, e por que nao e "ter modelo proprio":

a posicao de MVP exige que o produto consiga DEVOLVER uma inferencia defensavel
numa AOI. Isso se decompoe em cinco condicoes, e as duas ultimas sao as que
costumam ser esquecidas:

  1. as seis variaveis existem, na mesma cadeia e na mesma fonte de chuva
  2. existe modelo aplicavel -- proprio OU um pool que transfira para la
  3. ha rotulo suficiente para dizer alguma coisa
  4. o conjunto de teste tem CONTRASTE
  5. um modelo treinado FORA acerta la

A quarta este projeto ja aprendeu do jeito caro: um conjunto sem contraste
devolve 0,50 para qualquer modelo, e ai nao da para distinguir "o modelo
errou" de "nao havia o que acertar".

A quinta entrou depois da primeira versao desta matriz, que aprovou Curitiba
por engano. Curitiba passa na quarta -- separacao 0,239, acima do limiar -- e
mesmo assim o LOSO fica em 0,4997. Ter sinal interno e transferir sao coisas
diferentes, e a posicao de MVP exige a segunda. Sem a quinta condicao, a
matriz teria recomendado uma regiao onde o modelo nao funciona.

Por isso ela separa **aplicavel** de **validavel** de **transferivel**.

Por isso a matriz separa **aplicavel** de **validavel**. Petropolis e o caso
extremo util: tem terreno na mesma convencao das outras 123 derivacoes e
nenhum ponto rotulado. Da para APLICAR e nao da para VALIDAR. Anunciar uma
regiao assim como disponivel seria servir predicao sem saber o que ela vale.

NAO faz: nao treina, nao escreve no `region_registry.py`, nao promove regiao.
Produz a matriz; a decisao de mexer no registro e da Gabriela.

Uso:
    python scripts/suscetibilidade/mvp01_prontidao_por_regiao.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds03_esquema_alvo import VARIAVEIS_FISICAS, VERSAO  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
UNI = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.csv"
MEC02 = RUNS / "mod-mec-02" / "resultado.json"
PLUV01 = RUNS / "mod-pluv-01" / "resultado.json"
TER01 = RUNS / "ter-01-cadeia-harmonizada"
OUT = RUNS / "mvp-01-prontidao"

# separacao minima por feature isolada para o conjunto ser considerado com
# contraste. 0,20 em |2*(AUC-0,5)| equivale a AUC de 0,60 ou 0,40 -- abaixo
# disso nao da para distinguir "o modelo errou" de "nao havia o que acertar".
CONTRASTE_MINIMO = 0.20

# AUC minimo para dizer que um modelo treinado FORA funciona na regiao. 0,60 e
# o piso ja usado no projeto para distinguir sinal de acaso; abaixo disso a
# regiao pode receber predicao, mas ninguem pode afirmar que ela vale.
AUC_TRANSFERE_MINIMO = 0.60

# regioes sem ponto rotulado, mas com terreno derivado na mesma convencao
SO_TERRENO = {
    "petropolis": ("petropolis_harmonizado",
                   "serra tropical umida; N=0 rotulado (susc_20h). Terreno na "
                   "mesma convencao das outras derivacoes"),
}


def separacao(s: pd.DataFrame, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score

    fora = {}
    for f in VARIAVEIS_FISICAS:
        v = pd.to_numeric(s[f], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
            continue
        a = float(roc_auc_score(y[ok], v[ok]))
        fora[f] = round(abs(a - 0.5) * 2, 4)
    return fora


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"===MVP01_PRONTIDAO_POR_REGIAO=== esquema={VERSAO}")
    if not UNI.exists():
        print(f"ABORTADO: {UNI} ausente. Rode ds05.")
        return 1
    d = pd.read_csv(UNI, low_memory=False)

    loso = {}
    if MEC02.exists():
        m = json.loads(MEC02.read_text(encoding="utf-8"))
        for x in (m.get("variantes", {}).get("AMPLIADO", {})
                  .get("leave_one_source_out", []) or []):
            loso[x["estrato"]] = x.get("auc")
    else:
        print(f"AVISO: {MEC02} ausente -- sem validacao externa por fonte")

    linhas = []
    for fonte, s in d.groupby("fonte"):
        bin_ = s[s.classe.isin([0, 1])]
        y = bin_.classe.to_numpy().astype(int)
        seis = s[list(VARIAVEIS_FISICAS)].notna().all(axis=1)
        sep = separacao(bin_, y) if len(bin_) and len(np.unique(y)) > 1 else {}
        melhor = max(sep.values()) if sep else None

        linhas.append({
            "regiao": fonte,
            "n": int(len(s)),
            "pos": int((s.classe == 1).sum()),
            "neg": int((s.classe == 0).sum()),
            "grupos": int(s.grupo_cv.nunique()),
            "cadeia": "/".join(sorted(s.cadeia_terreno.unique())),
            "com_6_variaveis": int(seis.sum()),
            "pct_6_variaveis": round(100 * seis.mean(), 1),
            "fonte_chuva": "/".join(sorted(x for x in s.fonte_chuva.unique()
                                           if x != "ausente")) or "ausente",
            "mecanismo": "/".join(sorted(s.mecanismo.unique())),
            "nivel_negativo": "/".join(sorted(
                x for x in s.loc[s.classe == 0, "nivel_negativo"].unique())),
            "melhor_separacao": melhor,
            "loso_auc": loso.get(fonte),
        })

    for nome, (pasta, nota) in SO_TERRENO.items():
        tem = (TER01 / pasta / "run_manifest.json").exists()
        linhas.append({"regiao": nome, "n": 0, "pos": 0, "neg": 0, "grupos": 0,
                       "cadeia": "wbt30" if tem else "ausente",
                       "com_6_variaveis": 0, "pct_6_variaveis": 0.0,
                       "fonte_chuva": "ausente", "mecanismo": "FLUVIAL_ENXURRADA",
                       "nivel_negativo": "", "melhor_separacao": None,
                       "loso_auc": None, "nota": nota})

    t = pd.DataFrame(linhas)

    # ---- as quatro condicoes, avaliadas ----
    t["c1_variaveis"] = t.pct_6_variaveis >= 95
    t["c2_aplicavel"] = t.cadeia.str.contains("wbt30")
    t["c3_rotulo"] = (t.pos >= 30) & (t.neg >= 30)
    t["c4_contraste"] = t.melhor_separacao.fillna(0) >= CONTRASTE_MINIMO
    # C5 -- a condicao que faltava na primeira versao desta matriz, e que a
    # deixou aprovar Curitiba. Contraste interno acima do limiar NAO garante
    # que um modelo treinado fora acerte la: Curitiba tem separacao 0,239 e
    # LOSO de 0,4997, nivel de acaso. Ter sinal e transferir sao coisas
    # diferentes, e a posicao de MVP exige a segunda.
    t["c5_valida_fora"] = t.loso_auc.isna() | (t.loso_auc >= AUC_TRANSFERE_MINIMO)

    def veredito(r) -> str:
        if not r.c2_aplicavel:
            return "NAO_APLICAVEL_sem_cadeia_de_terreno"
        if not r.c1_variaveis:
            return "APLICAVEL_INCOMPLETO_faltam_variaveis"
        if not r.c3_rotulo:
            return "APLICAVEL_NAO_VALIDAVEL_sem_rotulo"
        if not r.c4_contraste:
            return "APLICAVEL_NAO_VALIDAVEL_sem_contraste"
        if not r.c5_valida_fora:
            return "APLICAVEL_MODELO_EXTERNO_NAO_TRANSFERE"
        if pd.isna(r.loso_auc):
            return "CANDIDATA_SEM_VALIDACAO_EXTERNA"
        return "CANDIDATA_A_MVP"

    t["veredito"] = t.apply(veredito, axis=1)

    print("\n--- MATRIZ DE PRONTIDAO ---")
    print(t[["regiao", "n", "pos", "neg", "pct_6_variaveis", "fonte_chuva",
             "melhor_separacao", "loso_auc", "veredito"]].to_string(index=False))

    print("\n--- AS CINCO CONDICOES ---")
    print(t[["regiao", "c1_variaveis", "c2_aplicavel", "c3_rotulo",
             "c4_contraste", "c5_valida_fora"]].to_string(index=False))

    cand = t[t.veredito == "CANDIDATA_A_MVP"]
    print(f"\n--- CANDIDATAS: {len(cand)} ---")
    for _, r in cand.iterrows():
        val = (f"validacao externa (treinado fora, testado la) = {r.loso_auc}"
               if r.loso_auc is not None else
               "sem validacao externa medida no pool")
        print(f"  {r.regiao}: contraste {r.melhor_separacao}, {val}")

    if PLUV01.exists():
        p = json.loads(PLUV01.read_text(encoding="utf-8"))
        c = p.get("comparacao") or {}
        if c:
            print(f"\n--- RECIFE, O MVP ATUAL ---")
            print(f"  o registro anuncia LOO-AUC {c['v12_loo_auc']}, medido sobre "
                  "chuva de fonte misturada")
            print(f"  com fonte unica: {c['agora_loo_auc']} "
                  f"({c['delta']:+.4f})")
            print(f"  {c['leitura']}")

    t.to_csv(OUT / "matriz_prontidao_mvp.csv", index=False)
    (OUT / "resumo.json").write_text(json.dumps({
        "esquema": VERSAO,
        "contraste_minimo": CONTRASTE_MINIMO,
        "condicoes": {
            "c1_variaveis": ">=95% das linhas com as seis variaveis fisicas",
            "c2_aplicavel": "terreno na cadeia wbt30",
            "c3_rotulo": ">=30 positivos e >=30 negativos",
            "c4_contraste": (f"melhor separacao por feature isolada >= "
                             f"{CONTRASTE_MINIMO}; abaixo disso nao da para "
                             "distinguir erro do modelo de ausencia de sinal"),
            "c5_valida_fora": (f"LOSO >= {AUC_TRANSFERE_MINIMO} quando existe. "
                               "Contraste interno nao garante transferencia"),
        },
        "matriz": json.loads(t.to_json(orient="records")),
        "candidatas": cand.regiao.tolist(),
        "nao_e": ("aplicavel nao e validavel: uma regiao com terreno e sem "
                  "rotulo pode receber predicao e nao pode confirma-la"),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
