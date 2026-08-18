"""
AUD-PROVENANCE-01 -- Toda coluna de procedencia auditada pelo mesmo teste.

DE ONDE VEIO:

o `aud_chuva01` mediu uma coisa especifica e achou um problema real: a coluna
de chuva de Recife carregava duas fontes e o INDICADOR de qual fonte produziu
o numero discriminava o rotulo melhor (0,8256) que a propria chuva (0,7377).
O teste era bom; o escopo e que era estreito. A mesma pergunta vale para
qualquer coluna de procedencia do contrato, e o `ds03` tem quatro delas.

Este script e aquele teste, parametrizado pela coluna em vez de fixo em chuva.

A GENERALIZACAO NAO E MECANICA, e ignorar isso daria um numero sem sentido:

`fonte_chuva` varia DENTRO das duas classes -- ha positivo e negativo em CHIRPS
e em ERA5. Ai a pergunta "a procedencia prediz o rotulo?" e legitima.

`nivel_negativo` nao. Ele so existe nos negativos; positivo recebe
`nao_aplicavel` por construcao. Perguntar se ele prediz o rotulo devolveria AUC
1,0 -- verdadeiro e vazio, porque a resposta esta na definicao e nao no dado.

Entao o script detecta em qual dos dois casos esta e troca a pergunta:

  MODO_ROTULO   a procedencia varia nas duas classes. Mede confundimento com o
                rotulo, distribuicao por estrato, e decompoe a discriminacao em
                (a) a variavel, (b) so o indicador de procedencia, (c) a
                variavel DENTRO de cada estrato, onde o indicador e constante.

  MODO_ESTRATO  a procedencia esta confinada a uma classe. Ai a pergunta e se
                os estratos sao FISICAMENTE DIFERENTES entre si: para cada par,
                o AUC de cada variavel separando um estrato do outro. Separacao
                alta significa que empilhar os dois produz uma classe
                heterogenea, e o modelo aprende parte da campanha que gerou o
                rotulo em vez do fenomeno.

POR QUE `nivel_negativo` E O PRIMEIRO ALVO:

a hierarquia do `ds01` ordena `observado > exclusao_qualificada > ausencia` e
diz que ausencia nao entra em modelo. Isso e regra declarada, nunca medida. E
a matriz do `mvp01` compara regioes cujo negativo vem de niveis diferentes sem
marcar isso em lugar nenhum -- Curitiba entra com ausencia, CEMS com observado,
UK com exclusao qualificada, e os tres viram uma coluna `neg` so. Se os niveis
forem fisicamente distintos, essa coluna nao e comparavel entre regioes.

NAO faz: nao corrige, nao remove linha, nao reordena a hierarquia. Mede.

Uso:
    python scripts/suscetibilidade/aud_provenance01_confundimento.py
    python scripts/suscetibilidade/aud_provenance01_confundimento.py --coluna fonte_chuva
    python scripts/suscetibilidade/aud_provenance01_confundimento.py --todas
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds03_esquema_alvo import (  # noqa: E402
    VARIAVEIS_FISICAS, VARIAVEIS_TERRENO, VERSAO,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
UNI = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.csv"
OUT = RUNS / "aud-provenance-01"

CHUVA = ("rain_max_24h", "rain_decay_index")
N_BOOT = 2000
SEMENTE = 20260816

# Cada coluna de procedencia com o valor que NAO e um estrato real -- o
# sentinela que significa "nao se aplica aqui" -- e as variaveis cuja
# interpretacao aquela procedencia afeta.
PROCEDENCIAS: dict[str, dict] = {
    "nivel_negativo": {
        "sentinela": ("nao_aplicavel",),
        "variaveis": VARIAVEIS_FISICAS,
        "porque": ("a hierarquia observado > exclusao_qualificada > ausencia e "
                   "declarada e nunca foi medida; o mvp01 compara regioes cujo "
                   "negativo vem de niveis diferentes sem marcar isso"),
    },
    "fonte_chuva": {
        "sentinela": ("ausente",),
        "variaveis": CHUVA,
        "porque": "reproduz o aud_chuva01 sob o teste generico",
    },
    "cadeia_terreno": {
        "sentinela": ("ausente",),
        "variaveis": VARIAVEIS_TERRENO,
        "porque": "global e wbt30 sao instrumentos diferentes, nao resolucoes",
    },
    "classe_relevo_base": {
        "sentinela": ("ausente",),
        "variaveis": VARIAVEIS_FISICAS,
        "porque": ("o criterio de relevo foi calibrado na janela do raster; "
                   "aplicado aos pontos e outro estimador"),
    },
}

PADRAO = "nivel_negativo"


def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """AUC por posto. Empates recebem posto medio, como manda o Mann-Whitney."""
    ok = np.isfinite(s) & np.isin(y, (0, 1))
    y, s = y[ok], s[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ic(y: np.ndarray, s: np.ndarray, n_boot: int = N_BOOT) -> dict:
    a = auc(y, s)
    if a is None:
        return {"auc": None, "n": 0, "motivo": "uma das classes esta vazia"}
    ok = np.isfinite(s) & np.isin(y, (0, 1))
    y, s = y[ok], s[ok]
    rng = np.random.default_rng(SEMENTE)
    amostras = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        v = auc(y[i], s[i])
        if v is not None:
            amostras.append(v)
    lo, hi = np.percentile(amostras, [2.5, 97.5]) if amostras else (None, None)
    return {"auc": round(a, 4),
            "ic95": [round(float(lo), 4), round(float(hi), 4)] if amostras else None,
            "separacao": round(abs(a - 0.5) * 2, 4),
            "n": int(len(y)), "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum())}


def escolher_modo(d: pd.DataFrame, coluna: str, sentinela) -> tuple[str, pd.DataFrame]:
    """A procedencia varia nas duas classes, ou esta presa a uma delas?

    Nao e configuracao: e lido do dado. Se todo estrato real cair numa classe
    so, perguntar se ele prediz o rotulo devolve a propria definicao.
    """
    real = d[~d[coluna].isin(sentinela) & d[coluna].notna()]
    if real.empty:
        return "SEM_ESTRATO", real
    classes = real["classe"].unique()
    if len(classes) > 1:
        return "MODO_ROTULO", d
    return "MODO_ESTRATO", real


def modo_rotulo(d: pd.DataFrame, coluna: str, variaveis, sentinela) -> dict:
    y = d["classe"].to_numpy()
    estratos = [f for f in d[coluna].dropna().unique() if f not in sentinela]
    rel: dict = {"modo": "MODO_ROTULO", "estratos": estratos,
                 "contagem": d[coluna].value_counts().to_dict()}

    rel["prevalencia_por_estrato"] = {
        str(f): round(float((s["classe"] == 1).mean()), 4)
        for f, s in d.groupby(coluna)}

    rel["discriminacao"] = {}
    for v in variaveis:
        bloco = {"global": auc_ic(y, d[v].to_numpy(dtype="float64"))}
        for f in estratos:
            s = d[d[coluna] == f]
            bloco[f"dentro__{f}"] = auc_ic(
                s["classe"].to_numpy(), s[v].to_numpy(dtype="float64"))
        rel["discriminacao"][v] = bloco

    if len(estratos) > 1:
        ind = (d[coluna] == estratos[0]).to_numpy(dtype="float64")
        rel["indicador_de_procedencia"] = auc_ic(y, ind)
        rel["veredito"] = "MISTURA_DE_PROCEDENCIA"
    else:
        rel["veredito"] = "PROCEDENCIA_UNICA"
    return rel


def modo_estrato(real: pd.DataFrame, coluna: str, variaveis) -> dict:
    """Os estratos sao fisicamente diferentes entre si?

    Um par com separacao alta significa que empilhar os dois produz uma classe
    heterogenea. Nao e erro por si -- e informacao que precisa ser reportada
    junto de qualquer numero calculado sobre a classe empilhada.
    """
    estratos = sorted(str(x) for x in real[coluna].dropna().unique())
    rel: dict = {"modo": "MODO_ESTRATO", "estratos": estratos,
                 "classe_de_origem": sorted(int(c) for c in real["classe"].unique()),
                 "contagem": {str(k): int(v) for k, v in
                              real[coluna].value_counts().items()}}
    if len(estratos) < 2:
        rel["veredito"] = "ESTRATO_UNICO"
        return rel

    pares = {}
    for a, b in itertools.combinations(estratos, 2):
        s = real[real[coluna].isin((a, b))]
        y = (s[coluna] == b).to_numpy().astype(int)
        por_var = {}
        for v in variaveis:
            por_var[v] = auc_ic(y, s[v].to_numpy(dtype="float64"))
        validos = [x["separacao"] for x in por_var.values()
                   if x.get("separacao") is not None]
        pares[f"{a} vs {b}"] = {
            "n_a": int((s[coluna] == a).sum()), "n_b": int((s[coluna] == b).sum()),
            "por_variavel": por_var,
            "melhor_separacao": round(max(validos), 4) if validos else None,
            "variavel_que_mais_separa": (
                max(por_var, key=lambda k: por_var[k].get("separacao") or 0)
                if validos else None),
        }
    rel["pares"] = pares
    piores = [k for k, v in pares.items() if (v["melhor_separacao"] or 0) >= 0.40]
    rel["pares_muito_distintos"] = piores
    rel["veredito"] = ("ESTRATOS_HETEROGENEOS" if piores
                       else "ESTRATOS_COMPATIVEIS")
    return rel


def auditar(d: pd.DataFrame, coluna: str, cfg: dict) -> dict:
    d = d[d["classe"].isin([0, 1])].copy()
    variaveis = [v for v in cfg["variaveis"] if v in d.columns]
    sentinela = cfg["sentinela"]
    modo, sub = escolher_modo(d, coluna, sentinela)
    if modo == "SEM_ESTRATO":
        return {"coluna": coluna, "modo": modo, "n": int(len(d)),
                "veredito": "SEM_ESTRATO_REAL"}
    if modo == "MODO_ROTULO":
        rel = modo_rotulo(sub, coluna, variaveis, sentinela)
    else:
        rel = modo_estrato(sub, coluna, variaveis)
    rel["coluna"] = coluna
    rel["n"] = int(len(sub))
    rel["porque"] = cfg["porque"]
    return rel


def imprimir(rel: dict, d: pd.DataFrame) -> None:
    col = rel["coluna"]
    print(f"\n{'='*70}\n[{col}] modo={rel['modo']} n={rel['n']:,} "
          f"veredito={rel.get('veredito')}")
    print(f"  motivo de auditar: {rel['porque']}")
    print(f"  contagem: {rel.get('contagem')}")

    if rel["modo"] == "MODO_ROTULO":
        print(f"  prevalencia de positivo por estrato: "
              f"{rel.get('prevalencia_por_estrato')}")
        for v, bloco in rel.get("discriminacao", {}).items():
            partes = [f"{k}={x['auc']}" for k, x in bloco.items()
                      if isinstance(x, dict) and x.get("auc") is not None]
            print(f"  AUC {v:20s} {'  '.join(partes)}")
        ind = rel.get("indicador_de_procedencia")
        if ind and ind.get("auc") is not None:
            print(f"  AUC do INDICADOR = {ind['auc']} IC95 {ind['ic95']}  "
                  f"<- isto nao e a variavel, e a procedencia")
        return

    print(f"  os estratos vivem so na classe {rel.get('classe_de_origem')}, "
          "entao a pergunta e se sao diferentes ENTRE SI")
    for par, x in rel.get("pares", {}).items():
        print(f"\n  {par}  (n={x['n_a']:,} vs {x['n_b']:,})")
        for v, y in x["por_variavel"].items():
            if y.get("auc") is None:
                continue
            marca = "  <--" if (y["separacao"] or 0) >= 0.40 else ""
            print(f"    {v:20s} AUC={y['auc']:.4f} IC95={y['ic95']} "
                  f"separacao={y['separacao']:.4f}{marca}")
        print(f"    => melhor separacao {x['melhor_separacao']} "
              f"em {x['variavel_que_mais_separa']}")

    # onde cada estrato vive, porque isso decide se a heterogeneidade e
    # confundida com regiao
    if "fonte" in d.columns:
        real = d[d[col].isin(rel["estratos"])]
        print(f"\n  onde cada estrato aparece:")
        print(pd.crosstab(real["fonte"], real[col]).to_string().replace("\n", "\n    "))


def main() -> int:
    args = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"===AUD_PROVENANCE01=== esquema={VERSAO}")
    if not UNI.exists():
        print(f"ABORTADO: {UNI} ausente. Rode ds05.")
        return 1
    d = pd.read_csv(UNI, low_memory=False)

    if "--todas" in args:
        alvos = list(PROCEDENCIAS)
    elif "--coluna" in args:
        alvos = [args[args.index("--coluna") + 1]]
    else:
        alvos = [PADRAO]

    resultado = {"esquema": VERSAO, "semente": SEMENTE, "n_bootstrap": N_BOOT,
                 "colunas": {}}
    for col in alvos:
        if col not in PROCEDENCIAS:
            print(f"coluna desconhecida: {col}. Conhecidas: {list(PROCEDENCIAS)}")
            return 2
        if col not in d.columns:
            print(f"coluna {col} ausente na tabela")
            continue
        rel = auditar(d, col, PROCEDENCIAS[col])
        resultado["colunas"][col] = rel
        imprimir(rel, d)

    heterog = [c for c, r in resultado["colunas"].items()
               if r.get("veredito") in ("ESTRATOS_HETEROGENEOS",
                                        "MISTURA_DE_PROCEDENCIA")]
    resultado["colunas_com_problema"] = heterog
    resultado["regra"] = (
        "numero calculado sobre uma classe que empilha estratos de procedencia "
        "diferentes precisa reportar a composicao. Nao e proibicao de empilhar: "
        "e proibicao de empilhar em silencio")

    print(f"\n{'='*70}\n--- VEREDITO ---")
    if heterog:
        for c in heterog:
            print(f"  {c}: {resultado['colunas'][c]['veredito']}")
        print("  Qualquer agregado sobre essas colunas tem de vir com a "
              "composicao ao lado.")
    else:
        print("  nenhuma coluna auditada mostrou estratos incompativeis")

    (OUT / "auditoria_procedencia.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    linhas = []
    for col, r in resultado["colunas"].items():
        for par, x in (r.get("pares") or {}).items():
            for v, y in x["por_variavel"].items():
                linhas.append({"coluna": col, "par": par, "variavel": v,
                               **{k: y.get(k) for k in
                                  ("auc", "ic95", "separacao", "n")}})
        for v, bloco in (r.get("discriminacao") or {}).items():
            for estrato, y in bloco.items():
                linhas.append({"coluna": col, "par": estrato, "variavel": v,
                               **{k: y.get(k) for k in
                                  ("auc", "ic95", "separacao", "n")}})
    if linhas:
        pd.DataFrame(linhas).to_csv(OUT / "auc_por_estrato.csv", index=False)
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
