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

Por isso ela separa **aplicavel**, **validavel** e **transferivel**. Petropolis e o caso
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

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
RUNS = REPO / "local_runs"
UNI = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.csv"
MEC03 = RUNS / "mod-mec-03" / "resultado.json"
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

# niveis que contam como negativo. Ausencia e lacuna de dado e fica de fora.
NEGATIVO_UTILIZAVEL = ("observado", "exclusao_qualificada")

# reamostragens do IC. Mesmo valor do aud_provenance01, semente propria fixa.
N_BOOT = 2000
SEMENTE = 20260816

# regioes sem ponto rotulado, mas com terreno derivado na mesma convencao
SO_TERRENO = {
    "petropolis": ("petropolis_harmonizado",
                   "serra tropical umida; N=0 rotulado (susc_20h). Terreno na "
                   "mesma convencao das outras derivacoes"),
}


def auc_posto(y: np.ndarray, s: np.ndarray) -> float | None:
    """AUC por posto, sem sklearn no laco de bootstrap -- 2.000 chamadas."""
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ic_bootstrap(y: np.ndarray, s: np.ndarray, grupos: np.ndarray | None = None,
                 n_boot: int = N_BOOT) -> list[float] | None:
    """IC95 percentil. Reamostra GRUPO quando ha grupo, e nao linha.

    POR QUE O GRUPO IMPORTA (Ploton et al., 2020, Nat. Commun. 11:4540): com
    dado espacialmente autocorrelacionado, reamostrar linha trata pontos
    vizinhos como observacoes independentes e estreita o intervalo
    artificialmente. O mesmo mecanismo que infla validacao cruzada aleatoria
    infla o IC. Reamostrar o grupo -- evento, AOI, chip -- preserva o bloco.
    """
    rng = np.random.default_rng(SEMENTE)
    amostras = []
    if grupos is None:
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            v = auc_posto(y[i], s[i])
            if v is not None:
                amostras.append(v)
    else:
        unicos = pd.unique(grupos)
        idx = {g: np.flatnonzero(grupos == g) for g in unicos}
        for _ in range(n_boot):
            esc = rng.choice(unicos, size=len(unicos), replace=True)
            linhas = np.concatenate([idx[g] for g in esc])
            v = auc_posto(y[linhas], s[linhas])
            if v is not None:
                amostras.append(v)
    if len(amostras) < 30:
        return None
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def separacao(s: pd.DataFrame, y: np.ndarray) -> tuple[dict, dict]:
    """Separacao por feature isolada, com IC no melhor valor.

    O IC vai so na melhor: e ela que entra na condicao C4, e bootstrapar as
    seis multiplicaria o custo por seis para informar o que nao decide nada.
    """
    fora, dados = {}, {}
    for f in VARIAVEIS_FISICAS:
        v = pd.to_numeric(s[f], errors="coerce").to_numpy(dtype="float64")
        ok = np.isfinite(v)
        if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
            continue
        a = auc_posto(y, v)
        if a is None:
            continue
        fora[f] = round(abs(a - 0.5) * 2, 4)
        dados[f] = v
    if not fora:
        return {}, {}
    melhor = max(fora, key=lambda k: fora[k])
    g = s["grupo_cv"].to_numpy() if "grupo_cv" in s.columns else None
    ic_auc = ic_bootstrap(y, dados[melhor], g)
    # o IC e do AUC; a separacao e |2*(AUC-0,5)|, monotona por parte. Converter
    # os dois extremos e reordenar da o intervalo da separacao.
    ic_sep = (sorted(round(abs(x - 0.5) * 2, 4) for x in ic_auc)
              if ic_auc else None)
    return fora, {"feature": melhor, "separacao": fora[melhor],
                  "auc": round(auc_posto(y, dados[melhor]), 4),
                  "ic95_auc": ic_auc, "ic95_separacao": ic_sep}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"===MVP01_PRONTIDAO_POR_REGIAO=== esquema={VERSAO}")
    if not UNI.exists():
        print(f"ABORTADO: {UNI} ausente. Rode ds05.")
        return 1
    d = pd.read_csv(UNI, low_memory=False)

    # o mec03 e preferido porque roda com as seis variaveis e ja traz IC; o
    # mec02 fica de reserva, e ai o IC vem vazio em vez de inventado
    loso, origem = {}, None
    if MEC03.exists():
        m = json.loads(MEC03.read_text(encoding="utf-8"))
        for x in m.get("leave_one_source_out", []) or []:
            loso[x["estrato"]] = {"auc": x.get("auc"), "ic95": x.get("ic95")}
        origem = "mod-mec-03 (seis variaveis, IC por bootstrap de grupo)"
    elif MEC02.exists():
        m = json.loads(MEC02.read_text(encoding="utf-8"))
        for x in (m.get("variantes", {}).get("AMPLIADO", {})
                  .get("leave_one_source_out", []) or []):
            loso[x["estrato"]] = {"auc": x.get("auc"), "ic95": None}
        origem = "mod-mec-02 (quatro variaveis, sem IC)"
    else:
        print("AVISO: nem mec03 nem mec02 -- sem validacao externa por fonte")
    if origem:
        print(f"validacao externa lida de: {origem}")

    linhas = []
    for fonte, s in d.groupby("fonte"):
        bin_ = s[s.classe.isin([0, 1])]
        y = bin_.classe.to_numpy().astype(int)
        seis = s[list(VARIAVEIS_FISICAS)].notna().all(axis=1)
        sep, melhor_det = (separacao(bin_, y)
                           if len(bin_) and len(np.unique(y)) > 1 else ({}, {}))
        melhor = melhor_det.get("separacao")

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
            "neg_utilizavel": int(s.loc[s.classe == 0, "nivel_negativo"]
                                  .isin(NEGATIVO_UTILIZAVEL).sum()),
            "melhor_separacao": melhor,
            "melhor_separacao_ic95": melhor_det.get("ic95_separacao"),
            "melhor_separacao_feature": melhor_det.get("feature"),
            "loso_auc": (loso.get(fonte) or {}).get("auc"),
            "loso_ic95": (loso.get(fonte) or {}).get("ic95"),
        })

    for nome, (pasta, nota) in SO_TERRENO.items():
        tem = (TER01 / pasta / "run_manifest.json").exists()
        linhas.append({"regiao": nome, "n": 0, "pos": 0, "neg": 0, "grupos": 0,
                       "cadeia": "wbt30" if tem else "ausente",
                       "com_6_variaveis": 0, "pct_6_variaveis": 0.0,
                       "fonte_chuva": "ausente", "mecanismo": "FLUVIAL_ENXURRADA",
                       "nivel_negativo": "", "neg_utilizavel": 0,
                       "melhor_separacao": None,
                       "melhor_separacao_ic95": None,
                       "melhor_separacao_feature": None,
                       "loso_auc": None, "loso_ic95": None, "nota": nota})

    t = pd.DataFrame(linhas)

    # ---- as quatro condicoes, avaliadas ----
    t["c1_variaveis"] = t.pct_6_variaveis >= 95
    t["c2_aplicavel"] = t.cadeia.str.contains("wbt30")
    # AUSENCIA DE REGISTRO NAO CONTA COMO NEGATIVO (decisao de 2026-08-16).
    #
    # Nao e negativo fraco -- e lacuna de dado. "Nao ha registro de enchente
    # aqui" e afirmacao sobre o REGISTRO, nao sobre o lugar: pode nao ter
    # inundado, pode ter inundado sem ninguem anotar. Tratar como classe 0
    # ensina ao modelo que o lugar e seguro por uma razao que e do arquivo.
    #
    # Isso ja estava na hierarquia do ds01 ("ausencia nao entra"), e agora esta
    # medido: o aud_provenance01 separa `ausencia` de `observado` com AUC 0,1913
    # em elevacao (separacao 0,617) e 0,7962 em chuva antecedente. Nao sao a
    # mesma populacao com confianca diferente, sao populacoes diferentes.
    #
    # `neg_utilizavel` e contado POR LINHA, na origem. A primeira versao deduzia
    # do texto agregado de niveis e dava o total inteiro para qualquer regiao com
    # ao menos um nivel utilizavel -- Curitiba aparecia com 442 quando so 114
    # tinham subido de ausencia. Erro de agregacao, nao de criterio.
    t["neg_e_lacuna"] = t.neg_utilizavel.eq(0) & t.neg.gt(0)
    t["c3_rotulo"] = (t.pos >= 30) & (t.neg_utilizavel >= 30)
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
        # Regiao sem NENHUM ponto e caso de rotulo, nao de variavel. Petropolis
        # tem terreno derivado na mesma convencao das outras 123 e aparecia como
        # "faltam variaveis" so porque a cobertura de 0 linhas da 0%. O motivo
        # verdadeiro -- nao ha o que cobrir -- ficava escondido atras de um
        # sintoma, e quem lesse a matriz depois concluiria que falta aquisicao
        # de variavel quando o que falta e evento rotulado.
        if r.n == 0:
            return "APLICAVEL_NAO_VALIDAVEL_sem_rotulo"
        if not r.c1_variaveis:
            return "APLICAVEL_INCOMPLETO_faltam_variaveis"
        if not r.c3_rotulo:
            if r.neg_e_lacuna:
                return "APLICAVEL_NAO_VALIDAVEL_negativo_e_lacuna_de_dado"
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
    # `nivel_negativo` passa a ser impresso, e nao so gravado no CSV. O
    # aud_provenance01 mediu que os tres niveis sao fisicamente distintos --
    # rain_max_24h separa exclusao de observado com AUC 0,8120 -- e que cada
    # nivel vive numa regiao diferente, sem sobreposicao. Comparar a coluna
    # `neg` entre regioes sem essa ao lado compara coisas diferentes.
    print(t[["regiao", "n", "pos", "neg", "neg_utilizavel", "nivel_negativo",
             "melhor_separacao", "melhor_separacao_ic95", "loso_auc",
             "loso_ic95", "veredito"]].to_string(index=False))
    print("  `neg` conta tudo; `neg_utilizavel` exclui ausencia de registro, "
          "que e lacuna de dado e nao observacao de que nao houve enchente")
    print("  cada nivel de negativo vive numa regiao so -- ver aud_provenance01")

    print("\n--- AS CINCO CONDICOES ---")
    print(t[["regiao", "c1_variaveis", "c2_aplicavel", "c3_rotulo",
             "c4_contraste", "c5_valida_fora"]].to_string(index=False))

    cand = t[t.veredito == "CANDIDATA_A_MVP"]
    print(f"\n--- CANDIDATAS: {len(cand)} ---")
    for _, r in cand.iterrows():
        val = (f"transferencia {r.loso_auc} IC95 {r.loso_ic95}"
               if r.loso_auc is not None and not pd.isna(r.loso_auc)
               else "sem validacao externa medida no pool")
        folga = ""
        if isinstance(r.loso_ic95, list) and r.loso_ic95:
            folga = (" -- LIMIAR DENTRO DO IC, aprovacao marginal"
                     if r.loso_ic95[0] < AUC_TRANSFERE_MINIMO
                     else " -- IC inteiro acima do limiar")
        print(f"  {r.regiao}: contraste {r.melhor_separacao} "
              f"IC95 {r.melhor_separacao_ic95} em {r.melhor_separacao_feature}, "
              f"{val}{folga}")

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
