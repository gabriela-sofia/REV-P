"""
AQ-CHIRPS3 v2 -- aquisicao robusta e resumivel da serie de chuva.

POR QUE UMA v2: a primeira versao trouxe 28 dos 97 meses do `rnl` e ZERO do
`sat`, com dezenas de `RasterioIOError`. Tres causas, atacadas aqui uma a uma:

  1. SEM RETRY POR DIA. Um erro de rede num dia matava o mes inteiro, e o mes
     nao era gravado. Agora cada dia tem 4 tentativas com espera crescente, e
     mes com dia faltando e gravado assim mesmo, com o buraco registrado.

  2. GDAL SEM RETRY HTTP. O /vsicurl aborta na primeira falha se
     GDAL_HTTP_MAX_RETRY nao estiver definido. Definido aqui antes de
     qualquer import de rasterio -- depois do import nao adianta.

  3. FLAVOR EM SERIE. `rnl` consumia todo o tempo e `sat` nunca comecava.
     Agora da para rodar um flavor por vez, e o padrao processa os meses em
     ordem de RELEVANCIA (os com mais eventos primeiro), para que uma
     interrupcao deixe a serie util em vez de deixar 1981-1990.

RESUMIVEL DE VERDADE: um mes so conta como pronto se o arquivo existe E tem
todos os dias esperados. Mes parcial e refeito. A v1 considerava pronto
qualquer arquivo existente, o que congelaria os buracos.

Uso:
    python scripts/externo/aq_chirps3_v2.py                 # rnl e sat
    python scripts/externo/aq_chirps3_v2.py --flavor rnl    # so um
    python scripts/externo/aq_chirps3_v2.py --status        # o que falta
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ANTES de importar rasterio/rioxarray: o GDAL le estas variaveis no import.
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
# NAO definir CPL_VSIL_CURL_USE_HEAD=NO. Diagnostico de 09/08/2026: essa
# variavel, que eu introduzi na v2 como "blindagem", QUEBRA o /vsicurl --
# sem a requisicao HEAD o GDAL nao descobre o tamanho do arquivo e nao
# consegue fazer as leituras por intervalo de bytes que o COG exige.
# Foi a causa unica das 2.594 falhas e das 163 minutos de espera atribuidas
# a "bloqueio por scraping". O servidor sempre respondeu HTTP 200.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
STORE = RUNS / "geostore"
OUT = RUNS / "aq-chirps3"
AOI_BNG = (350_000, 350_000, 400_000, 450_000)
BBOX = (-3.25, 52.60, -1.50, 54.40)
DIAS_ANTES = 14
THREADS = 4
PAUSA = 0.15
TENTATIVAS = 4


def url_dia(d: dt.date, flavor: str) -> str:
    return ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/"
            f"{flavor}/cogs/{d.year}/chirps-v3.0.{flavor}.{d.year}."
            f"{d.month:02d}.{d.day:02d}.cog")


def datas_necessarias() -> list[dt.date]:
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from geo_store import ler_bbox

    g = ler_bbox(STORE / "outlines_elegiveis.parquet", AOI_BNG)
    d = pd.to_datetime(g["start_date"], errors="coerce", utc=True).dt.date.dropna()
    precisa: set = set()
    for x in sorted(set(d)):
        for k in range(DIAS_ANTES + 1):
            precisa.add(x - dt.timedelta(days=k))
    return sorted(precisa)


def peso_dos_meses(datas: list[dt.date]) -> Counter:
    """Quantas DATAS DE EVENTO cada mes contem -- define a ordem de prioridade."""
    c: Counter = Counter()
    for d in datas:
        c[(d.year, d.month)] += 1
    return c


def buscar_dia(d: dt.date, flavor: str):
    import numpy as np
    import rioxarray

    ultimo = None
    for tentativa in range(TENTATIVAS):
        try:
            da = rioxarray.open_rasterio(url_dia(d, flavor), chunks=None,
                                         masked=True, lock=False)
            da = da.rio.clip_box(minx=BBOX[0], miny=BBOX[1],
                                 maxx=BBOX[2], maxy=BBOX[3]).load()
            da = da.where(da != -9999.0)
            ds = da.to_dataset(name="precip").squeeze("band", drop=True)
            return ds.expand_dims(time=[np.datetime64(d)])
        except Exception as exc:  # noqa: BLE001
            ultimo = exc
            time.sleep(1.5 * (tentativa + 1))
    raise RuntimeError(f"{d} {type(ultimo).__name__}")


def mes_completo(caminho: Path, esperado: int) -> bool:
    """Mes so conta como pronto se tem TODOS os dias esperados."""
    if not caminho.exists() or caminho.stat().st_size == 0:
        return False
    try:
        import xarray as xr

        with xr.open_dataset(caminho) as ds:
            return int(ds.sizes.get("time", 0)) >= esperado
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    args = sys.argv[1:]
    flavors = ["rnl", "sat"]
    if "--flavor" in args:
        flavors = [args[args.index("--flavor") + 1]]

    datas = datas_necessarias()
    por_mes: dict = defaultdict(list)
    for d in datas:
        por_mes[(d.year, d.month)].append(d)
    peso = peso_dos_meses(datas)
    # meses com mais datas de evento primeiro: interromper deixa o util pronto
    ordem = sorted(por_mes, key=lambda m: (-peso[m], m))

    if "--status" in args:
        print("===AQ_CHIRPS3_V2_STATUS===")
        print(f"DATAS={len(datas)} MESES={len(por_mes)}")
        for flavor in ("rnl", "sat"):
            d = OUT / flavor
            prontos = sum(1 for m in ordem if mes_completo(
                d / f"chirps3_{flavor}_{m[0]}-{m[1]:02d}.nc", len(por_mes[m])))
            print(f"[{flavor}] completos={prontos}/{len(ordem)}")
        print("===END===")
        return 0

    print("===AQ_CHIRPS3_V2===")
    print(f"DATAS={len(datas)} MESES={len(por_mes)} FLAVORS={flavors}")
    import xarray as xr

    relatorio = {}
    for flavor in flavors:
        dir_f = OUT / flavor
        dir_f.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        feitos = pulados = 0
        falhas: list[str] = []

        for i, m in enumerate(ordem, 1):
            dias = sorted(por_mes[m])
            destino = dir_f / f"chirps3_{flavor}_{m[0]}-{m[1]:02d}.nc"
            if mes_completo(destino, len(dias)):
                pulados += 1
                continue
            partes = []
            with ThreadPoolExecutor(max_workers=THREADS) as ex:
                jobs = {}
                for d in dias:
                    jobs[d] = ex.submit(buscar_dia, d, flavor)
                    time.sleep(PAUSA)
                for d in dias:
                    try:
                        partes.append(jobs[d].result())
                    except Exception as exc:  # noqa: BLE001
                        falhas.append(str(exc))
            if partes:
                # grava mesmo incompleto: dado parcial vale mais que nenhum,
                # e o buraco fica registrado em `falhas`
                xr.concat(partes, dim="time").to_netcdf(destino)
                feitos += 1
            if i % 10 == 0:
                print(f"  [{flavor}] {i}/{len(ordem)} meses "
                      f"(novos={feitos} cache={pulados} falhas={len(falhas)}) "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)

        arquivos = sorted(dir_f.glob("*.nc"))
        completos = sum(1 for m in ordem if mes_completo(
            dir_f / f"chirps3_{flavor}_{m[0]}-{m[1]:02d}.nc", len(por_mes[m])))
        tam = sum(p.stat().st_size for p in arquivos) / 1024 ** 2
        print(f"[{flavor}] meses_completos={completos}/{len(ordem)} "
              f"arquivos={len(arquivos)} {tam:.1f}MB falhas_dia={len(falhas)} "
              f"tempo={(time.time()-t0)/60:.1f}min")
        for f in falhas[:8]:
            print(f"   FALHA {f}")
        relatorio[flavor] = {"meses_completos": completos, "meses_alvo": len(ordem),
                             "arquivos": len(arquivos), "mb": round(tam, 1),
                             "falhas_dia": len(falhas), "exemplos": falhas[:20]}

    (OUT / "aquisicao_v2.json").write_text(json.dumps({
        "datas": len(datas), "meses": len(por_mes), "bbox": BBOX,
        "ordem": "meses com mais datas de evento primeiro",
        "relatorio": relatorio,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
