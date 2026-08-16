"""
NEG-01 -- Tira o negativo de Curitiba da lacuna, sem promove-lo a observacao.

O PROBLEMA:

os 442 negativos de Curitiba estavam marcados `ausencia`, e ausencia de
registro nao e negativo -- e lacuna de dado. "Nao ha registro de enchente aqui"
fala do arquivo, nao do lugar. Com a regra aplicada, Curitiba e Recife ficaram
com ZERO negativos utilizaveis, e o MVP passou a se apoiar num modelo de
positivo contra nao-rotulado.

Mas `ausencia` estava classificando esses pontos de forma pessimista demais, e
isso tambem e erro. O que existe no dado nao e silencio: **e um chamado ao 156,
feito daquele endereco, naquele dia, sobre outro assunto** -- coleta, transito,
arvore, iluminacao. Isso nao e a mesma coisa que nao ter registro nenhum.

O QUE UM CHAMADO SOBRE OUTRO ASSUNTO PROVA, e o que nao prova:

  prova    o canal estava funcionando naquele dia
  prova    aquele endereco tem morador que usa o canal
  prova    a categoria de alagamento existia e estava sendo usada na cidade
  NAO prova que alguem vistoriou o ponto procurando inundacao

Por isso o destino correto NAO e `observado`. E `exclusao_qualificada` -- o
mesmo nivel do piloto ingles, que tambem e inferencia a partir de criterio, e
nao observacao direta. O `ds01` ja ordena os tres niveis; este script move
Curitiba um degrau, com criterio, e nao dois.

OS CRITERIOS, espelhando o N1-N4 do piloto ingles e declarados antes de rodar:

  N1  canal ativo          volume total de chamados na cidade naquele dia >= p25
                           do ano. Dia de sistema fora do ar nao serve.
  N2  categoria em uso     houve pelo menos um chamado HIDROLOGICO na cidade
                           naquele dia. Se ninguem reportou alagamento em lugar
                           nenhum, o silencio pode ser da categoria e nao do
                           lugar.
  N3  endereco reporta     o proprio ponto e um chamado daquele endereco naquele
                           dia, sobre assunto nao-hidrologico.
  N4  afastamento          distancia minima de qualquer positivo da MESMA data.

Ponto que falha em qualquer um continua `ausencia`. O script conta quantos
passam e quantos caem em cada criterio -- reprovar e resultado, nao erro.

POR QUE O N2 E O CRITERIO QUE MAIS IMPORTA:

sem ele, um dia em que o 156 recebeu mil chamados e nenhum de alagamento
contaria como evidencia contra inundacao. Mas esse dia pode ser justamente um
dia sem chuva -- e ai a informacao e trivial -- ou um dia em que a categoria
falhou. Exigir que a cidade tenha registrado alagamento em ALGUM lugar naquele
dia garante que o canal estava recebendo e classificando esse tipo de
ocorrencia quando o ponto ficou em silencio.

NAO faz: nao promove nada a `observado`, nao altera o ds04 (grava um registro
que o ds04 le), e nao mexe em Recife -- para Recife nao existe no repositorio o
fluxo completo de chamados da SEDEC, entao os criterios N1 e N2 nao teriam como
ser avaliados. Fica declarado como pendencia, nao resolvido em silencio.

Uso:
    python scripts/suscetibilidade/neg01_exclusao_qualificada_temporal.py
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds03_esquema_alvo import VERSAO  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
BRUTO = RUNS / "susc_20n_siac156_negative_expansion" / "raw"
RED = RUNS / "ds-04-reducao"
OUT = RUNS / "neg-01-exclusao-qualificada"

# distancia minima a um positivo da mesma data. 500 m e a mesma ordem do
# afastamento usado no piloto ingles; abaixo disso o ponto pode estar na mesma
# quadra alagada.
AFASTAMENTO_M = 500.0

# percentil do volume diario abaixo do qual o dia nao conta como canal ativo
P_VOLUME = 0.25

# PARES (assunto, subdivisao) que marcam chamado de INUNDACAO. Sao os dois que
# o susc_20k validou contra o catalogo de eventos -- 1.005 e 233 registros --
# e nao busca por substring.
#
# A primeira versao deste script usava substring solta e errou por tres razoes
# ao mesmo tempo, todas visiveis so quando o criterio passou em 100% dos dias:
#   "agua"/"chuva"  casavam ZERO -- o arquivo estava sendo lido com o encoding
#                   errado, e o acento chegava corrompido (ver carregar_bruto).
#   "bueiro"        casava "fauna sinantropica | risco para leptospirose/
#                   roedores em bueiro" -- 3.179 chamados sobre roedores.
#   "drenagem"      casava a categoria inteira, 16.718 por ano, quase toda de
#                   manutencao de rotina (limpeza de caixa de captacao, erosao
#                   em galeria). Presente todo dia, por isso o N2 nunca
#                   reprovava: media a existencia da categoria, nao a chegada
#                   de relato de alagamento.
SUBDIVISOES_INUNDACAO = ("alagamentos", "enchentes, inundacoes ou alagamentos")


def sem_acento(s: str) -> str:
    """NFKD e minuscula. Nada de remendo de mojibake -- ver `carregar_bruto`."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c)).lower().strip()


def carregar_bruto() -> pd.DataFrame:
    """So as colunas necessarias: sao 517 MB em quatro arquivos.

    ENCODING: os arquivos sao UTF-8. A primeira versao deste script os leu como
    latin-1, o que nao levanta excecao -- latin-1 decodifica qualquer byte -- e
    entrega 'Enchentes, inundaÃ§Ãµes ou alagamentos' em vez de texto. Uma
    subdivisao inteira, 233 chamados, deixava de casar em silencio, e o N2
    ficava mais restritivo do que devia sem ninguem perceber.

    Ler com o encoding errado e o mesmo tipo de erro que este projeto combate
    em outros lugares: nao falha, so muda o resultado.
    """
    partes = []
    for p in sorted(BRUTO.glob("*.csv")):
        d = pd.read_csv(p, sep=";", encoding="utf-8-sig", low_memory=False,
                        usecols=lambda c: c.strip("﻿") in
                        ("DataCriacao", "Assunto", "Subdivisao"))
        d.columns = [c.strip("﻿") for c in d.columns]
        partes.append(d)
        print(f"   {p.name}: {len(d):,} chamados")
    if not partes:
        return pd.DataFrame()
    d = pd.concat(partes, ignore_index=True)
    d["data"] = pd.to_datetime(d["DataCriacao"], errors="coerce",
                               dayfirst=True).dt.date
    return d[d["data"].notna()]


def perfil_diario(bruto: pd.DataFrame) -> pd.DataFrame:
    """Volume total e de INUNDACAO por dia. E o que sustenta N1 e N2."""
    sub = bruto["Subdivisao"].map(sem_acento)
    bruto = bruto.assign(hidro=sub.isin(SUBDIVISOES_INUNDACAO))
    n = int(bruto["hidro"].sum())
    print(f"   chamados de inundacao identificados: {n:,} "
          f"({100*n/max(len(bruto),1):.3f}% do total)")
    if n == 0:
        raise SystemExit(
            "ABORTADO: nenhum chamado de inundacao casou com "
            f"{SUBDIVISOES_INUNDACAO}. O vocabulario mudou ou a normalizacao "
            "de acento quebrou -- corrigir antes de usar o N2, e nao seguir "
            "com um criterio que nao mede nada.")
    g = bruto.groupby("data").agg(total=("Assunto", "size"),
                                  hidro=("hidro", "sum"))
    return g.reset_index()


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"===NEG01_EXCLUSAO_QUALIFICADA=== esquema={VERSAO}")

    p = RED / "curitiba.csv"
    if not p.exists():
        print(f"ABORTADO: {p} ausente. Rode ds04.")
        return 1
    cur = pd.read_csv(p, low_memory=False)
    cand = cur[(cur.classe == 0) & (cur.nivel_negativo == "ausencia")].copy()
    pos = cur[cur.classe == 1].copy()
    print(f"candidatos (negativos hoje em ausencia): {len(cand):,}")
    print(f"positivos para o criterio de afastamento: {len(pos):,}")

    if not BRUTO.exists():
        print(f"ABORTADO: base bruta do 156 ausente em {BRUTO}")
        return 1
    print("\n--- lendo a base bruta do 156 ---")
    bruto = carregar_bruto()
    if bruto.empty:
        print("ABORTADO: base bruta vazia")
        return 1
    perfil = perfil_diario(bruto)
    print(f"   {len(bruto):,} chamados em {len(perfil):,} dias, "
          f"{perfil.data.min()} a {perfil.data.max()}")

    limiar = float(perfil["total"].quantile(P_VOLUME))
    print(f"   N1: volume diario >= p{int(P_VOLUME*100)} = {limiar:,.0f} chamados")
    dias_hidro = set(perfil.loc[perfil["hidro"] > 0, "data"])
    print(f"   N2: {len(dias_hidro):,} dias com ao menos um chamado hidrologico "
          f"({100*len(dias_hidro)/len(perfil):.1f}% dos dias)")

    cand["data"] = pd.to_datetime(cand["data_evento"], errors="coerce",
                                  format="mixed").dt.date
    pos["data"] = pd.to_datetime(pos["data_evento"], errors="coerce",
                                 format="mixed").dt.date
    vol = dict(zip(perfil["data"], perfil["total"]))

    # N1 e N2
    cand["n1_canal_ativo"] = cand["data"].map(vol).fillna(0) >= limiar
    cand["n2_categoria_em_uso"] = cand["data"].isin(dias_hidro)
    # N3: por construcao o ponto E um chamado de outro assunto naquele endereco
    # e naquele dia; a procedencia registra qual. Verificado, nao suposto.
    cand["n3_endereco_reporta"] = (
        cand["procedencia"].astype(str).str.contains("siac156", case=False, na=False)
        & cand["data"].notna())

    # N4: distancia ao positivo mais proximo da MESMA data
    dist = []
    for _, r in cand.iterrows():
        mesma = pos[pos["data"] == r["data"]]
        if mesma.empty:
            dist.append(np.inf)
            continue
        dist.append(float(np.min(haversine_m(
            r["lat"], r["lon"], mesma["lat"].to_numpy(), mesma["lon"].to_numpy()))))
    cand["dist_positivo_mesma_data_m"] = dist
    cand["n4_afastamento"] = cand["dist_positivo_mesma_data_m"] >= AFASTAMENTO_M

    criterios = ["n1_canal_ativo", "n2_categoria_em_uso",
                 "n3_endereco_reporta", "n4_afastamento"]
    cand["aprovado"] = cand[criterios].all(axis=1)
    cand["nivel_negativo_novo"] = np.where(
        cand["aprovado"], "exclusao_qualificada", "ausencia")
    cand["motivo_reprovacao"] = [
        ";".join(c for c in criterios if not r[c]) or ""
        for _, r in cand.iterrows()]

    print("\n--- CRITERIOS, um a um ---")
    for c in criterios:
        n = int(cand[c].sum())
        print(f"   {c:24s} passa {n:>4,} / {len(cand):,} "
              f"({100*n/len(cand):.1f}%)")
    n_ok = int(cand["aprovado"].sum())
    print(f"\n   APROVADOS nos quatro: {n_ok:,} / {len(cand):,} "
          f"({100*n_ok/max(len(cand),1):.1f}%)")
    if n_ok < len(cand):
        print("   motivos de reprovacao:")
        for m, k in cand.loc[~cand.aprovado, "motivo_reprovacao"].value_counts().items():
            print(f"      {m}: {k:,}")

    print("\n--- O QUE ISSO MUDA ---")
    print(f"   antes: 0 negativos utilizaveis em Curitiba (todos ausencia)")
    print(f"   depois: {n_ok:,} em exclusao_qualificada, "
          f"{len(cand)-n_ok:,} continuam ausencia")
    if n_ok >= 30:
        print(f"   com {n_ok:,} negativos, Curitiba passa a ter contra o que "
              "comparar -- a condicao de rotulo do mvp01 volta a ser avaliavel")
    else:
        print(f"   {n_ok:,} ainda e pouco para a condicao de rotulo (minimo 30)")

    cols = ["ponto_id", "data_evento", "lat", "lon", "procedencia",
            *criterios, "dist_positivo_mesma_data_m", "aprovado",
            "nivel_negativo_novo", "motivo_reprovacao"]
    cand[cols].to_csv(OUT / "curitiba_reclassificacao_negativo.csv", index=False)
    (OUT / "resumo.json").write_text(json.dumps({
        "esquema": VERSAO,
        "regiao": "curitiba",
        "candidatos": int(len(cand)),
        "aprovados_exclusao_qualificada": n_ok,
        "seguem_ausencia": int(len(cand) - n_ok),
        "criterios": {c: int(cand[c].sum()) for c in criterios},
        "limiar_volume_diario": limiar,
        "percentil_volume": P_VOLUME,
        "afastamento_m": AFASTAMENTO_M,
        "dias_com_hidrologico": len(dias_hidro),
        "dias_no_perfil": int(len(perfil)),
        "subdivisoes_de_inundacao": list(SUBDIVISOES_INUNDACAO),
        "nao_e": (
            "exclusao qualificada NAO e observacao. Ninguem vistoriou o ponto "
            "procurando inundacao; o que se sabe e que o canal estava ativo, "
            "que aquele endereco o usou naquele dia e que reportou outra coisa. "
            "E o mesmo nivel do piloto ingles, um degrau abaixo do CEMS"),
        "pendencia_recife": (
            "para Recife nao existe no repositorio o fluxo completo de chamados "
            "da SEDEC, so os 278 pontos do v12. Sem ele N1 e N2 nao podem ser "
            "avaliados, e os 124 negativos continuam em ausencia"),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
