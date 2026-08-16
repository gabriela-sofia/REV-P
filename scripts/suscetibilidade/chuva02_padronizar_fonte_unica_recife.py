"""
CHUVA-02 -- Padroniza os 278 pontos de Recife numa unica fonte de chuva.

DE ONDE VEIO ISTO:
`aud_chuva01_fontes_incompativeis.py` mediu, nao supos: a coluna de chuva de
Recife carrega duas fontes na mesma coluna, 181 pontos em CHIRPS v2.0 e 97 em
Open-Meteo/ERA5-Land, e o indicador de fonte tem AUC de rank (0,826) maior que
a propria chuva (0,738) contra o rotulo. Como a chuva antecedente e o
preditor principal da trilha pluvial (coeficiente +0,9896, p < 1e-4, contra
HAND em -0,0001, p = 0,978), parte do que aquele coeficiente mede nao e chuva:
e proveniencia. O proprio audit script diz explicitamente o que NAO faz:
"reamostrar CHIRPS para os 97 pontos de ERA5-Land e aquisicao de dado, nao
auditoria". Este script e essa aquisicao -- na direcao inversa.

POR QUE A DIRECAO E OPEN-METEO PARA TODOS, E NAO CHIRPS PARA TODOS:
as duas direcoes tem custo, de natureza diferente.
  - CHIRPS para os 97: o servidor da CHC (data.chc.ucsb.edu) tem protecao
    documentada contra scraping -- `aq_chirps3_v3.py` registra uma tentativa
    de 163 minutos com quase zero sucesso antes de existir cache por dia. Os
    97 pontos cobrem 85 datas distintas entre 2015 e 2022 (nao sao 2 eventos
    concentrados), o que exigiria uma campanha dia-a-dia sem garantia de
    terminar.
  - Open-Meteo para os 181: a mesma API ja foi usada com sucesso 3 vezes
    neste projeto (v8, v9 bairro-matched, v12 lead A/C) sem historico de
    bloqueio, com uma requisicao por PONTO (nao por dia), entao 181 pontos e
    181 requisicoes -- minutos, nao horas.
Decisao tomada com a Gabriela em 2026-08-16: Open-Meteo/ERA5-Land para os 278.

O QUE ISTO NAO RESOLVE, declarado: ERA5-Land e reanalise (modelo), nao produto
com estacao real como o CHIRPS gauge-blend. Perde-se a validacao por
pluviometro que a maioria dos pontos tinha. Isso e custo, nao e escondido.

METODO, identico ao ja usado nas fontes Open-Meteo existentes (mesma janela,
mesma formula, para nao introduzir uma SEGUNDA inconsistencia dentro da fonte
unica): janela de 14 dias antes do evento, indice de decaimento com fator
0,85/dia, maximo de chuva em 24h dentro da janela.

O QUE ISTO NAO FAZ: nao toca PROJETO (so le point_id/lat/lon/event_date de
`dataset_v12_final.csv`, read-only); nao decide sobre o v12 em si; nao reroda
ds04/ds05/mod_mec02 -- isso e o proximo passo, separado.

Uso:
    python scripts/suscetibilidade/chuva02_padronizar_fonte_unica_recife.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
PROJETO = REPO.parent / "PROJETO"
RUNS = REPO / "local_runs"

V12 = PROJETO / "local_runs" / "recife_modelo_v12_extracao_final" / "dataset_v12_final.csv"
HARM = RUNS / "ter-03-brasil-harmonizado" / "recife_harmonizado.csv"
OUT = RUNS / "chuva-02-fonte-unica-recife"

FONTE_ALVO = "open_meteo_era5_land_archive_api"
FONTE_ORIGEM_A_TROCAR = "chirps_v2.0_gauge_satellite_blend"

DECAY_K = 0.85
LOOKBACK = 14
# concorrencia moderada -- o mesmo padrao ja usado com sucesso em
# fetch_rain_v9_bairro_matched.py (PROJETO), 6 workers, sem historico de
# bloqueio nesta API neste projeto
N_WORKERS = 4


def fetch_um(lat: float, lon: float, event_date: pd.Timestamp) -> dict:
    if pd.isna(event_date):
        return {"rain_max_24h": np.nan, "rain_decay_index_api": np.nan,
                "n_dias_encontrados": 0, "erro": "event_date ausente na origem"}
    start = (event_date - pd.Timedelta(days=LOOKBACK)).date()
    end = (event_date - pd.Timedelta(days=1)).date()
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           "&daily=precipitation_sum&timezone=America%2FRecife")
    # 429 e limite de taxa, nao ausencia de dado -- backoff e reenvio, ate 4
    # tentativas, antes de declarar falha de verdade
    r = None
    for tentativa in range(4):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 429:
                time.sleep(2.0 * (tentativa + 1))
                continue
            r.raise_for_status()
            break
        except Exception as e:  # noqa: BLE001
            if tentativa == 3:
                return {"rain_max_24h": np.nan, "rain_decay_index_api": np.nan,
                        "n_dias_encontrados": 0, "erro": str(e)}
            time.sleep(1.5 * (tentativa + 1))
    else:
        return {"rain_max_24h": np.nan, "rain_decay_index_api": np.nan,
                "n_dias_encontrados": 0, "erro": "429 apos 4 tentativas"}
    j = r.json()
    daily = j.get("daily", {})
    serie = dict(zip(daily.get("time", []), daily.get("precipitation_sum", [])))
    ordenado = [serie.get(str((event_date - pd.Timedelta(days=k)).date()))
                for k in range(LOOKBACK, 0, -1)]
    arr = np.array([np.nan if v is None else float(v) for v in ordenado])
    n_encontrados = int(np.sum(~np.isnan(arr)))
    if n_encontrados == 0:
        return {"rain_max_24h": np.nan, "rain_decay_index_api": np.nan,
                "n_dias_encontrados": 0, "erro": None}
    rain_max = float(np.nanmax(arr))
    preenchido = np.where(np.isnan(arr), 0.0, arr)
    api = sum(p * (DECAY_K ** (LOOKBACK - idx - 1))
              for idx, p in enumerate(preenchido))
    return {"rain_max_24h": round(rain_max, 2),
            "rain_decay_index_api": round(float(api), 3),
            "n_dias_encontrados": n_encontrados, "erro": None}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===CHUVA02_PADRONIZAR_FONTE_UNICA_RECIFE===")

    if not V12.exists():
        print(f"ABORTADO: {V12} ausente.")
        return 1
    if not HARM.exists():
        print(f"ABORTADO: {HARM} ausente. Rode ter03_reextrair_brasil.py.")
        return 1

    v12 = pd.read_csv(V12, low_memory=False)
    v12["event_date"] = pd.to_datetime(v12["event_date"])
    alvo = v12[v12["rain_data_source"] == FONTE_ORIGEM_A_TROCAR].copy()
    print(f"PONTOS_V12={len(v12)} A_REAMOSTRAR={len(alvo)} "
          f"(hoje em '{FONTE_ORIGEM_A_TROCAR}')")

    linhas = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fetch_um, r.lat, r.lon, r.event_date): r.point_id
                for r in alvo.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            res["point_id"] = futs[fut]
            linhas.append(res)
            if i % 30 == 0:
                print(f"  ... {i}/{len(alvo)}  ({time.time()-t0:.0f}s)", flush=True)

    novo = pd.DataFrame(linhas)
    falhas = novo[novo["rain_max_24h"].isna()]
    print(f"REAMOSTRADOS={len(novo)}  FALHAS={len(falhas)}  "
          f"tempo={(time.time()-t0):.0f}s")
    if len(falhas):
        print("  pontos sem dado (ficam NaN, nao estimados):",
              falhas["point_id"].tolist())

    # ---- aplica no harmonizado: sobrescreve so os 181, mantem os 97 como estao ----
    h = pd.read_csv(HARM, low_memory=False)
    antes = h["rain_data_source"].value_counts().to_dict()

    m = h["point_id"].isin(novo["point_id"])
    mapa_max = dict(zip(novo["point_id"], novo["rain_max_24h"]))
    mapa_api = dict(zip(novo["point_id"], novo["rain_decay_index_api"]))
    h.loc[m, "rain_max_24h_chirps"] = h.loc[m, "point_id"].map(mapa_max)
    h.loc[m, "rain_decay_index_api_chirps"] = h.loc[m, "point_id"].map(mapa_api)
    h.loc[m, "rain_data_source"] = FONTE_ALVO

    depois = h["rain_data_source"].value_counts().to_dict()
    print(f"\nFONTE_CHUVA antes={antes}")
    print(f"FONTE_CHUVA depois={depois}")

    ainda_sem = h[h["rain_max_24h_chirps"].isna()]
    print(f"AINDA_SEM_CHUVA={len(ainda_sem)} de {len(h)} "
          f"(NaN declarado, nao preenchido)")

    h.to_csv(HARM, index=False)
    novo.to_csv(OUT / "reamostragem_open_meteo_181pts.csv", index=False)
    (OUT / "resumo.json").write_text(json.dumps({
        "fonte_alvo": FONTE_ALVO,
        "pontos_reamostrados": int(len(novo)),
        "falhas": int(len(falhas)),
        "point_id_falhas": falhas["point_id"].tolist(),
        "fonte_chuva_antes": antes,
        "fonte_chuva_depois": depois,
        "ainda_sem_chuva": int(len(ainda_sem)),
        "metodo": {"lookback_dias": LOOKBACK, "decay_k": DECAY_K,
                   "formula": "rain_max_24h = max(precip diario na janela); "
                              "rain_decay_index_api = soma(precip_i * "
                              "decay_k**(dias_antes_do_evento_i - 1))"},
        "arquivo_alterado": str(HARM),
        "nao_toca": "PROJETO e read-only; so leu point_id/lat/lon/event_date",
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={HARM} (sobrescrito)")
    print(f"GRAVADO={OUT}/reamostragem_open_meteo_181pts.csv")
    print("===END===")
    return 0 if len(falhas) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
