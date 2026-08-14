"""
DS-04 -- Reduz cada fonte, uma por vez, ao esquema-alvo congelado no ds03.

O QUE ESTE SCRIPT E, E O QUE ELE DELIBERADAMENTE NAO E:

e uma traducao. Cada fonte tem um redutor proprio, isolado dos demais, que
sabe apenas duas coisas: as colunas daquela fonte e o contrato do ds03. Nenhum
redutor le o resultado de outro. Isso e o que permite conferir um por vez e o
que impede que uma convencao de Recife vaze para Curitiba sem ninguem ver.

NAO e um filtro. Nada e descartado aqui, nem mesmo linha invalida: a decisao
de admitir e do `ds05`, que registra o motivo de cada rejeicao. Separar
traducao de admissao e o que torna o relatorio de contagem conferivel -- se as
duas coisas acontecessem juntas, "sumiu um ponto" nao teria resposta.

DECISOES DE TRADUCAO QUE PRECISAM ESTAR DECLARADAS, porque nenhuma e obvia:

  RECIFE entra em cadeia NATIVA de 10 m, nao harmonizada a 30 m. Nao e
  desleixo: Recife e pluvial urbano, o coeficiente de HAND la e -0,0001 com
  p = 0,978, e a decisao fechada em ext_resolucao_e_mecanismo_decisao_v1.md e
  que a resolucao segue o MECANISMO. Harmonizar Recife a 30 m custaria a
  declividade (7,20 graus caem para 2,65) para refinar uma variavel que o
  proprio modelo diz nao usar. A variante a 30 m e gravada em arquivo separado
  para quem quiser testar o pool -- fora da tabela unica, por decisao.

  CURITIBA entra em cadeia wbt30, porque e fluvial e o pool fluvial exige a
  mesma derivacao em todas as regioes.

  O NEGATIVO DE CURITIBA e declarado `ausencia`, nao `observado`. O campo de
  origem diz `siac156_curitiba_non_hydrological_category_v1`: existe um chamado
  no 156 e o assunto nao era hidrologico. Isso nao e "a area foi analisada e
  nao havia inundacao" -- e ausencia de registro de inundacao onde ha registro
  de outra coisa. Promover isso a negativo observado seria o erro que a
  hierarquia do ds01 existe para impedir.

  CLASSE DE RELEVO tem tres procedencias diferentes e a coluna
  `classe_relevo_base` diz qual: medida na janela do raster (CEMS, criterio
  original do cems03), calculada nos pontos amostrados (UK), ou declarada pelo
  perfil da regiao (Recife, Curitiba). O criterio -- relevo local >= 400 m E
  fracao acima de 15 graus >= 0,25 -- foi calibrado na janela. Aplicado aos
  pontos e outro estimador, e por isso fica marcado.

  SEN1FLOODS11 E UFO nao tem declividade nem TWI, e isso e do ds01: sao
  derivadas direcionais que exigem grade local reprojetada, nao calculada para
  661 chips em 6 continentes. Ficam nulas declaradas. Sem slope nao ha como
  calcular fracao acima de 15 graus, entao o relevo fica NAO_CLASSIFICADO.

  PETROPOLIS nao aparece. Tem derivacao de terreno (ter01) e nao tem nenhum
  ponto rotulado -- N = 0, ver susc_20h. Entra no relatorio como fonte de
  contagem zero, para que a ausencia seja visivel em vez de silenciosa.

Uso:
    python scripts/suscetibilidade/ds04_reduzir_por_fonte.py
    python scripts/suscetibilidade/ds04_reduzir_por_fonte.py --fonte recife
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds03_esquema_alvo import (  # noqa: E402
    CLASSE, COLUNAS, VARIAVEIS_TERRENO, VERSAO,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
OUT = RUNS / "ds-04-reducao"

DS01 = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
HARM_TODAS = RUNS / "ter-02-comparacao" / "dataset_harmonizado_todas.csv"
HARM_SERRA = RUNS / "ter-02-comparacao" / "dataset_serra_harmonizado.csv"
HARM_UK = RUNS / "ter-02-comparacao" / "dataset_harmonizado_uk.csv"
HARM_BR = RUNS / "ter-03-brasil-harmonizado"
RELEVO_AOI = RUNS / "cems-02-analogos-v2" / "verificacao_relevo_por_aoi.csv"

FLUVIAL = "FLUVIAL_ENXURRADA"
PLUVIAL = "PLUVIAL_URBANO"
MISTO = "MISTO_NAO_SEPARAVEL"

# criterio do cems03, repetido aqui porque e o criterio, nao um detalhe
RELEVO_LOCAL_MIN_M = 400.0
FRAC_GT15_MIN = 0.25

MAPA_FONTE_CHUVA = {
    "chirps_v2.0_gauge_satellite_blend": "chirps_v2_gauge_satellite_blend",
    "open_meteo_era5_land_archive_api": "open_meteo_era5_land",
}


def moldar(d: pd.DataFrame) -> pd.DataFrame:
    """Garante as colunas do contrato, na ordem, sem inventar valor."""
    for c in COLUNAS:
        if c not in d.columns:
            d[c] = np.nan
    d["classe_nome"] = d["classe"].map(CLASSE)
    return d[list(COLUNAS)].copy()


def nivel_negativo(classe: pd.Series, quando_zero: str) -> pd.Series:
    return pd.Series(
        np.where(classe == 0, quando_zero, "nao_aplicavel"), index=classe.index)


def relevo_dos_pontos(elev: pd.Series, slope: pd.Series) -> tuple[str, dict]:
    """Mesmo criterio do cems03, aplicado a amostra. Estimador diferente."""
    e = pd.to_numeric(elev, errors="coerce").dropna()
    s = pd.to_numeric(slope, errors="coerce").dropna()
    if len(e) < 30 or len(s) < 30:
        return "NAO_CLASSIFICADO", {"motivo": "amostra insuficiente ou sem slope"}
    # p2-p98 em vez de min-max: um outlier de amostragem nao decide o estrato
    relevo = float(e.quantile(0.98) - e.quantile(0.02))
    frac = float((s > 15).mean())
    cls = ("INGREME" if relevo >= RELEVO_LOCAL_MIN_M and frac >= FRAC_GT15_MIN
           else "PLANO_OU_ONDULADO")
    return cls, {"relevo_local_p2p98_m": round(relevo, 1),
                 "frac_gt15": round(frac, 4), "n_elev": len(e), "n_slope": len(s)}


def sobrepor_terreno_harmonizado(d: pd.DataFrame) -> pd.DataFrame:
    """Substitui as quatro de terreno pela cadeia wbt30 onde ela existir.

    A coluna `cadeia_terreno` passa a wbt30 SO nas linhas em que as quatro
    chegaram juntas. Linha com tres de wbt30 e uma de global seria a mistura
    que a cadeia harmonizada existe para impedir.
    """
    # Duas origens de sobreposicao, porque sao duas chaves diferentes: o CEMS
    # casa por AOI (ter02) e o piloto ingles casa por raster unico (ter05).
    # Concatenar aqui e o que evita um redutor especial para o UK.
    base = HARM_TODAS if HARM_TODAS.exists() else HARM_SERRA
    caminhos = [p for p in (base, HARM_UK) if p.exists()]
    d["cadeia_terreno"] = "global"
    if not caminhos:
        print(f"   AVISO: nenhuma sobreposicao harmonizada em {base.parent}")
        return d
    h = pd.concat([pd.read_csv(p, low_memory=False) for p in caminhos],
                  ignore_index=True)
    cols = {f: f"{f}_wbt" for f in VARIAVEIS_TERRENO if f"{f}_wbt" in h.columns}
    if not cols or "ponto_id" not in h.columns:
        print("   AVISO: sobreposicao sem colunas _wbt utilizaveis")
        return d
    origem = " + ".join(p.name for p in caminhos)
    d = d.merge(h[["ponto_id", *cols.values()]].drop_duplicates("ponto_id"),
                on="ponto_id", how="left")
    completo = d[list(cols.values())].notna().all(axis=1)
    for f, c in cols.items():
        d.loc[completo, f] = d.loc[completo, c]
    d.loc[completo, "cadeia_terreno"] = "wbt30"
    d = d.drop(columns=list(cols.values()))
    print(f"   terreno: {int(completo.sum()):,} wbt30 / "
          f"{int((~completo).sum()):,} global  (de {origem})")
    return d


# --------------------------------------------------------------------------
# redutores -- um por fonte, sem dependencia entre eles
# --------------------------------------------------------------------------

def _ds01(regiao_filtro) -> pd.DataFrame:
    d = pd.read_csv(DS01, low_memory=False)
    return d[regiao_filtro(d["regiao"])].copy()


def reduzir_uk() -> tuple[pd.DataFrame, dict]:
    d = _ds01(lambda r: r == "UK_noroeste")
    if d.empty:
        return pd.DataFrame(columns=COLUNAS), {"nota": "ds01 sem UK_noroeste"}
    d = d.rename(columns={"rain_decay_index_api": "rain_decay_index"})
    d["fonte"] = "uk"
    d["aoi"] = "UK_noroeste_piloto"
    d["procedencia"] = d["fonte_label"]
    d["resolucao_nativa_m"] = 30.0
    # ft_uk04 amostra CHIRPS v3, flavor rnl. Nao e o mesmo produto do CHIRPS
    # v2 usado em Recife, e por isso tem valor proprio no dominio.
    d["fonte_chuva"] = np.where(d["rain_max_24h"].notna(), "chirps_v3_rnl", "ausente")
    d["nivel_negativo"] = nivel_negativo(d["classe"], "exclusao_qualificada")
    d["unidade_agrupamento"] = "evento"
    d["mecanismo"] = FLUVIAL
    d = sobrepor_terreno_harmonizado(d)
    cls, diag = relevo_dos_pontos(d["elevation_m"], d["slope_deg"])
    d["classe_relevo"] = cls
    d["classe_relevo_base"] = ("pontos_amostrados" if cls != "NAO_CLASSIFICADO"
                               else "ausente")
    return moldar(d), {"relevo": {"classe": cls, **diag}}


def reduzir_cems() -> tuple[pd.DataFrame, dict]:
    d = _ds01(lambda r: r.astype(str).str.startswith("analogo_EMSR"))
    if d.empty:
        return pd.DataFrame(columns=COLUNAS), {"nota": "ds01 sem ativacoes CEMS"}
    d = d.rename(columns={"rain_decay_index_api": "rain_decay_index"})
    d["fonte"] = "cems"
    # no CEMS o grupo de validacao E a AOI: e onde o analista desenhou.
    d["aoi"] = d["grupo_cv"].astype(str)
    d["procedencia"] = "copernicus_ems_observado"
    d["resolucao_nativa_m"] = pd.to_numeric(d["resolucao_m"], errors="coerce")
    d["fonte_chuva"] = "ausente"
    d["nivel_negativo"] = nivel_negativo(d["classe"], "observado")
    d["unidade_agrupamento"] = "aoi"
    d["mecanismo"] = FLUVIAL
    d = sobrepor_terreno_harmonizado(d)

    d["classe_relevo"] = "NAO_CLASSIFICADO"
    d["classe_relevo_base"] = "ausente"
    diag: dict = {}
    if RELEVO_AOI.exists():
        rel = pd.read_csv(RELEVO_AOI)
        mapa = dict(zip(rel["aoi"].astype(str), rel["classe_relevo"].astype(str)))
        achou = d["aoi"].map(mapa)
        d.loc[achou.notna(), "classe_relevo"] = achou[achou.notna()]
        d.loc[achou.notna(), "classe_relevo_base"] = "janela_raster_cems03"
        diag = {"aois_com_relevo": int(achou.notna().sum()),
                "aois_sem_relevo": sorted(d.loc[achou.isna(), "aoi"].unique().tolist())[:10],
                "por_classe": d["classe_relevo"].value_counts().to_dict()}
    return moldar(d), {"relevo": diag}


def _reduzir_nivel1(regiao: str, fonte: str, mecanismo: str,
                    ) -> tuple[pd.DataFrame, dict]:
    d = _ds01(lambda r: r == regiao)
    if d.empty:
        return pd.DataFrame(columns=COLUNAS), {"nota": f"ds01 sem {regiao}"}
    d = d.rename(columns={"rain_decay_index_api": "rain_decay_index"})
    d["fonte"] = fonte
    d["aoi"] = d["grupo_cv"].astype(str)
    d["procedencia"] = d["fonte_label"]
    d["resolucao_nativa_m"] = pd.to_numeric(d["resolucao_m"], errors="coerce")
    d["fonte_chuva"] = "ausente"
    d["nivel_negativo"] = nivel_negativo(d["classe"], "observado")
    d["unidade_agrupamento"] = "chip"
    d["mecanismo"] = mecanismo
    d = sobrepor_terreno_harmonizado(d)
    # sem slope nao existe fracao acima de 15 graus: o criterio nao e aplicavel
    d["classe_relevo"] = "NAO_CLASSIFICADO"
    d["classe_relevo_base"] = "ausente"
    return moldar(d), {"relevo": {"motivo": "sem slope_deg no ds01 (declarado la)"}}


def reduzir_sen1floods11():
    return _reduzir_nivel1("nivel1_sen1floods11_handlabeled",
                           "sen1floods11", FLUVIAL)


def reduzir_ufo():
    return _reduzir_nivel1("nivel1_ufo_urban_flood_observations", "ufo", MISTO)


def _reduzir_brasil(nome: str, cfg: dict, sufixo_terreno: str,
                    cadeia: str) -> tuple[pd.DataFrame, dict]:
    p = HARM_BR / f"{nome}_harmonizado.csv"
    if not p.exists():
        return pd.DataFrame(columns=COLUNAS), {"nota": f"{p.name} ausente; rode ter03"}
    b = pd.read_csv(p, low_memory=False)

    d = pd.DataFrame(index=b.index)
    # O identificador da fonte NAO e unico em Curitiba: 150 `point_id` aparecem
    # duas vezes, mesma coordenada, mesma unidade de observacao, mesmo rotulo --
    # a pseudo-replicacao ja conhecida do SIAC 156. O contrato exige ponto_id
    # unico, entao a repeticao ganha ordinal em vez de ser deduplicada aqui:
    # descartar linha e decisao de admissao, e admissao e do ds05.
    bruto = b["point_id"].astype(str)
    n_repetidos = int(bruto.duplicated().sum())
    ordinal = bruto.groupby(bruto).cumcount()
    d["ponto_id"] = (f"{nome}__" + bruto
                     + np.where(ordinal > 0, "#" + ordinal.astype(str), ""))
    d["fonte"] = nome
    d["aoi"] = cfg["aoi"]
    d["classe"] = pd.to_numeric(b["label"], errors="coerce").astype("Int64")
    d["grupo_cv"] = b[cfg["grupo"]].astype(str)
    d["unidade_agrupamento"] = cfg["unidade"]
    d["resolucao_nativa_m"] = cfg["resolucao_nativa_m"]
    d["cadeia_terreno"] = cadeia
    d["mecanismo"] = cfg["mecanismo"]
    d["classe_relevo"] = cfg["classe_relevo"]
    d["classe_relevo_base"] = "declarada"
    d["lat"] = b["lat"]
    d["lon"] = b["lon"]
    d["data_evento"] = b.get("event_date")

    # procedencia carrega o rotulo de origem EXATO, para nada se perder na
    # traducao: e por ele que se reencontra a linha no pipeline da fonte.
    origem = b.get("negative_source_type")
    papel = b.get("dataset_role_source")
    d["procedencia"] = np.where(
        d["classe"] == 0,
        origem.fillna("negativo_sem_tipo_declarado") if origem is not None
        else "negativo_sem_tipo_declarado",
        papel.fillna(cfg["procedencia_positivo"]) if papel is not None
        else cfg["procedencia_positivo"])
    d["nivel_negativo"] = nivel_negativo(d["classe"], cfg["nivel_negativo"])

    for alvo, col in zip(VARIAVEIS_TERRENO, cfg["colunas_terreno"]):
        d[alvo] = pd.to_numeric(b[col], errors="coerce") if col in b.columns else np.nan
    d["rain_max_24h"] = pd.to_numeric(b.get("rain_max_24h_chirps"), errors="coerce")
    d["rain_decay_index"] = pd.to_numeric(
        b.get("rain_decay_index_api_chirps"), errors="coerce")
    src = b.get("rain_data_source")
    d["fonte_chuva"] = (src.map(MAPA_FONTE_CHUVA).fillna("ausente")
                        if src is not None else "ausente")

    diag = {"n": int(len(d)), "grupos": int(d["grupo_cv"].nunique()),
            "point_id_repetido_na_origem": n_repetidos,
            "fonte_chuva": d["fonte_chuva"].value_counts().to_dict(),
            "nivel_negativo": d["nivel_negativo"].value_counts().to_dict()}
    if len(d["fonte_chuva"].unique()) > 1:
        diag["ALERTA"] = ("mais de uma fonte de chuva na mesma fonte de dados; "
                          "ver aud_chuva01")
    return moldar(d), diag


CFG_RECIFE = {
    "aoi": "recife_rmr",
    "grupo": "point_id",
    "unidade": "ponto",
    "resolucao_nativa_m": 10.0,
    "mecanismo": PLUVIAL,
    # planicie costeira; o proprio v12 registra terreno de 5 a 8 m sobre a
    # drenagem. Declarada, nao medida por janela de raster.
    "classe_relevo": "PLANO_OU_ONDULADO",
    "procedencia_positivo": "sedec_recife_ocorrencia_v12",
    # negativo do Protocolo C: nao ha registro da SEDEC naquele ponto. Isso e
    # ausencia, e a hierarquia do ds01 e explicita em que ausencia nao entra
    # como negativo observado.
    "nivel_negativo": "ausencia",
    "colunas_terreno": ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf"],
}

CFG_RECIFE_WBT30 = {**CFG_RECIFE,
                    "colunas_terreno": ["elevation_m_wbt30", "slope_deg_wbt30",
                                        "hand_m_dinf_wbt30", "twi_dinf_wbt30"]}

CFG_CURITIBA = {
    "aoi": "curitiba_amc",
    "grupo": "observation_unit_key",
    "unidade": "unidade_observacao",
    "resolucao_nativa_m": 10.0,
    "mecanismo": FLUVIAL,
    # planalto: relevo local ~100 m contra 1.152 m de Petropolis (cems03)
    "classe_relevo": "PLANO_OU_ONDULADO",
    "procedencia_positivo": "siac156_curitiba_ocorrencia_hidrologica",
    "nivel_negativo": "ausencia",
    "colunas_terreno": ["elevation_m_wbt30", "slope_deg_wbt30",
                        "hand_m_dinf_wbt30", "twi_dinf_wbt30"],
}


def reduzir_recife():
    return _reduzir_brasil("recife", CFG_RECIFE, "", "nativa_10m")


def reduzir_recife_variante_wbt30():
    return _reduzir_brasil("recife", CFG_RECIFE_WBT30, "_wbt30", "wbt30")


def reduzir_curitiba():
    return _reduzir_brasil("curitiba", CFG_CURITIBA, "_wbt30", "wbt30")


REDUTORES = {
    "uk": reduzir_uk,
    "cems": reduzir_cems,
    "sen1floods11": reduzir_sen1floods11,
    "ufo": reduzir_ufo,
    "recife": reduzir_recife,
    "curitiba": reduzir_curitiba,
}

# nao entram na tabela unica; existem para auditoria e teste de pool
VARIANTES = {"recife__variante_wbt30": reduzir_recife_variante_wbt30}

# fonte conhecida, sem nenhum ponto. Fica no relatorio para que a ausencia
# seja um numero e nao um esquecimento.
SEM_PONTOS = {
    "petropolis": ("tem derivacao de terreno no ter01 e nenhum ponto rotulado "
                   "(N=0, susc_20h); serra tropical umida, mecanismo fluvial/"
                   "enxurrada declarado"),
}


def main() -> int:
    args = sys.argv[1:]
    alvo = args[args.index("--fonte") + 1] if "--fonte" in args else None
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"===DS04_REDUZIR_POR_FONTE=== esquema={VERSAO}")

    if not DS01.exists():
        print(f"ABORTADO: {DS01} ausente. Rode ds01.")
        return 1

    relatorio: dict = {"esquema": VERSAO, "fontes": {}, "sem_pontos": SEM_PONTOS}
    for nome, fn in {**REDUTORES, **VARIANTES}.items():
        if alvo and nome != alvo:
            continue
        print(f"\n[{nome}]")
        d, diag = fn()
        if d.empty:
            print(f"   VAZIO -- {diag.get('nota', 'sem motivo declarado')}")
            relatorio["fontes"][nome] = {"n": 0, **diag}
            continue
        d.to_csv(OUT / f"{nome}.csv", index=False)
        info = {
            "n": int(len(d)),
            "grupos": int(d["grupo_cv"].nunique()),
            "classes": {CLASSE.get(k, k): int(v) for k, v in
                        d["classe"].value_counts().items()},
            "nivel_negativo": d["nivel_negativo"].value_counts().to_dict(),
            "cadeia_terreno": d["cadeia_terreno"].value_counts().to_dict(),
            "fonte_chuva": d["fonte_chuva"].value_counts().to_dict(),
            "classe_relevo": d["classe_relevo"].value_counts().to_dict(),
            "terreno_completo": int(d[list(VARIAVEIS_TERRENO)].notna().all(axis=1).sum()),
            "com_chuva": int(d["rain_max_24h"].notna().sum()),
            **diag,
        }
        relatorio["fontes"][nome] = info
        print(f"   n={info['n']:,} grupos={info['grupos']:,} "
              f"terreno_completo={info['terreno_completo']:,} "
              f"com_chuva={info['com_chuva']:,}")
        print(f"   classes={info['classes']} negativo={info['nivel_negativo']}")
        print(f"   cadeia={info['cadeia_terreno']} relevo={info['classe_relevo']}")

    for nome, motivo in SEM_PONTOS.items():
        print(f"\n[{nome}] N=0 -- {motivo}")

    (OUT / "relatorio_reducao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
