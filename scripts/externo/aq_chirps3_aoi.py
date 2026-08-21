"""
AQ-CHIRPS3 -- chuva diaria na AOI, por leitura de janela em COG.

DESCOBERTA QUE TORNA ISSO BARATO (2026-08-07):
os arquivos diarios do CHIRPS v3 sao Cloud Optimized GeoTIFF. Isso permite ler
APENAS a janela da AOI por HTTP, com range requests, em vez de baixar o arquivo
global inteiro. A AOI do piloto, a 0,05 grau, tem cerca de 27 x 28 pixels --
uns 3 KB por dia contra ~30 MB do arquivo completo. Os 1.297 dias necessarios
saem em megabytes.

O template correto foi obtido da implementacao de referencia do projeto
dhis2/dhis2eo (`dhis2eo/data/chc/chirps3/daily.py`), depois de quatro palpites
de URL falharem na sondagem. Os erros dos palpites foram: faltava o nivel
`final/`, faltava `cogs/`, a extensao e `.cog` e nao `.tif.gz`, e o nome do
arquivo carrega o flavor.

  https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/{flavor}/cogs/
      {ano}/chirps-v3.0.{flavor}.{ano}.{mes}.{dia}.cog

DOIS FLAVORS, e usar os dois e decisao metodologica, nao capricho:
  rnl -- particao do total pentadal usando ERA5   (primaria, recomendada)
  sat -- particao do total pentadal usando IMERG  (analise de sensibilidade)

A avaliacao de 24 produtos de precipitacao em 18.428 bacias (HESS, 2026)
mostrou CHIRPS e ERA5-Land subestimando extremos, com IMERG-F melhor na
deteccao de pico. Como `rain_max_24h` e uma feature de extremo, rodar as duas
particoes e a forma barata de medir se essa escolha muda a conclusao, em vez
de assumir que nao muda.

LIMITACAO QUE FICA REGISTRADA: o v12 brasileiro usa CHIRPS v2.0, que cobre
apenas 50S-50N e nao alcanca a Inglaterra (50-56N). Nao existe paridade
possivel de fonte entre as duas regioes -- v3 e a familia mais proxima, mas e
outra versao. Isso e limitacao declarada, nao defeito corrigivel.

O servidor da CHC tem protecao contra scraping. A implementacao de referencia
usa no maximo 5 threads e 0,3 s entre submissoes; isso e respeitado aqui.

NAO faz: nao extrai feature, nao calcula indice de decaimento, nao treina.
So adquire a serie diaria recortada na AOI.

Uso:
    python scripts/externo/aq_chirps3_aoi.py --sondar
    python scripts/externo/aq_chirps3_aoi.py --baixar
    python scripts/externo/aq_chirps3_aoi.py --baixar --flavor sat
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
RUNS = REPO / "local_runs"
STORE = RUNS / "geostore"
OUT = RUNS / "aq-chirps3"
AOI_BNG = (350_000, 350_000, 400_000, 450_000)
# bbox em WGS84 com folga de 0,15 grau, para nao perder pixel de borda
BBOX = (-3.25, 52.60, -1.50, 54.40)
DIAS_ANTES = 14
THREADS = 5
PAUSA = 0.3


def url_dia(d: dt.date, flavor: str) -> str:
    return (
        "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/"
        f"{flavor}/cogs/{d.year}/chirps-v3.0.{flavor}.{d.year}.{d.month:02d}.{d.day:02d}.cog"
    )


def datas_necessarias() -> list[dt.date]:
    """Datas de evento na AOI mais a janela anterior, lidas do geostore."""
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


def ler_janela(url: str):
    import rioxarray

    da = rioxarray.open_rasterio(url, chunks=None, masked=True, lock=False)
    da = da.rio.clip_box(minx=BBOX[0], miny=BBOX[1], maxx=BBOX[2], maxy=BBOX[3])
    return da.load()


def buscar_dia(d: dt.date, flavor: str):
    import numpy as np

    da = ler_janela(url_dia(d, flavor))
    da = da.where(da != -9999.0)
    ds = da.to_dataset(name="precip").squeeze("band", drop=True)
    ds = ds.expand_dims(time=[np.datetime64(d)])
    ds["precip"].attrs.update(long_name="Precipitation", units="mm/day")
    return ds


def modo_sondar() -> int:
    print("===AQ_CHIRPS3_SONDAGEM===")
    datas = datas_necessarias()
    print(f"DATAS_NECESSARIAS={len(datas)} ({datas[0]} a {datas[-1]}, evento+{DIAS_ANTES}d)")
    print(f"BBOX={BBOX}")
    teste = datas[len(datas) // 2]
    for flavor in ("rnl", "sat"):
        t0 = time.time()
        try:
            ds = buscar_dia(teste, flavor)
            forma = dict(ds.sizes)
            n = int(ds["precip"].notnull().sum())
            print(f"[{flavor}] OK dia={teste} forma={forma} pixels_validos={n} "
                  f"em {time.time()-t0:.1f}s")
            print(f"   url={url_dia(teste, flavor)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{flavor}] FALHOU {type(exc).__name__}: {str(exc)[:140]}")
    meses = len({(d.year, d.month) for d in datas})
    print(f"MESES_A_GRAVAR={meses}")
    print(f"ESTIMATIVA_TEMPO~{len(datas)*PAUSA/60:.0f}min por flavor (limite do servidor)")
    print("===END===")
    return 0


def modo_baixar(flavors: list[str]) -> int:
    import xarray as xr

    OUT.mkdir(parents=True, exist_ok=True)
    datas = datas_necessarias()
    por_mes: dict[tuple[int, int], list[dt.date]] = defaultdict(list)
    for d in datas:
        por_mes[(d.year, d.month)].append(d)

    print("===AQ_CHIRPS3===")
    print(f"DATAS={len(datas)} MESES={len(por_mes)} FLAVORS={flavors}")
    relatorio = {}

    for flavor in flavors:
        dir_f = OUT / flavor
        dir_f.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        feitos, pulados, falhas = 0, 0, []

        for (ano, mes), dias in sorted(por_mes.items()):
            destino = dir_f / f"chirps3_{flavor}_{ano}-{mes:02d}.nc"
            if destino.exists() and destino.stat().st_size > 0:
                pulados += 1
                continue
            with ThreadPoolExecutor(max_workers=THREADS) as ex:
                jobs = {}
                for d in dias:
                    jobs[d] = ex.submit(buscar_dia, d, flavor)
                    time.sleep(PAUSA)
                partes = []
                for d in dias:
                    try:
                        partes.append(jobs[d].result())
                    except Exception as exc:  # noqa: BLE001
                        falhas.append(f"{flavor} {d} {type(exc).__name__}")
            if not partes:
                continue
            xr.concat(partes, dim="time").to_netcdf(destino)
            feitos += 1
            if feitos % 20 == 0:
                print(f"  [{flavor}] {feitos} meses gravados, "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)

        arquivos = sorted(dir_f.glob("*.nc"))
        bytes_tot = sum(p.stat().st_size for p in arquivos)
        print(f"[{flavor}] meses_novos={feitos} ja_existentes={pulados} "
              f"arquivos={len(arquivos)} {bytes_tot/1024**2:.1f}MB "
              f"falhas={len(falhas)} tempo={(time.time()-t0)/60:.1f}min")
        for f in falhas[:10]:
            print(f"   FALHA {f}")
        relatorio[flavor] = {
            "meses_gravados": len(arquivos), "bytes": bytes_tot,
            "falhas": falhas, "minutos": round((time.time() - t0) / 60, 1),
        }

    (OUT / "aquisicao.json").write_text(
        json.dumps({"datas": len(datas), "meses": len(por_mes),
                    "bbox": BBOX, "relatorio": relatorio},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / "PROVENIENCIA.md").write_text(
        "# Proveniencia - CHIRPS v3 diario, recorte da AOI EXT-UK\n"
        "- Fonte: Climate Hazards Center, UC Santa Barbara\n"
        "- Produto: CHIRPS v3.0 daily, stage=final, flavors=rnl (ERA5) e sat (IMERG)\n"
        "- Acesso: leitura de janela em COG por HTTP (nao download do global)\n"
        f"- Data de acesso: {time.strftime('%Y-%m-%d %H:%M')}\n"
        f"- BBox: {BBOX} (WGS84, folga de 0,15 grau)\n"
        f"- Datas: {len(datas)} dias = data de evento + {DIAS_ANTES} dias anteriores\n"
        "- Natureza: OBSERVADO/estimado por satelite+estacao. E FEATURE, nunca rotulo.\n"
        "- LIMITACAO DECLARADA: o v12 brasileiro usa CHIRPS v2.0, que cobre so\n"
        "  50S-50N e nao alcanca a Inglaterra. Nao ha paridade de versao possivel\n"
        "  entre as duas regioes; v3 e a familia mais proxima. Declarar como\n"
        "  limitacao, nao corrigir em silencio.\n"
        "- Status: AQUISICAO. Nenhuma feature calculada, nenhum indice de\n"
        "  decaimento derivado, nenhum modelo treinado.\n",
        encoding="utf-8")
    print("===END===")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if "--sondar" in args:
        return modo_sondar()
    if "--baixar" in args:
        if "--flavor" in args:
            return modo_baixar([args[args.index("--flavor") + 1]])
        return modo_baixar(["rnl", "sat"])
    print("uso: --sondar | --baixar [--flavor rnl|sat]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
