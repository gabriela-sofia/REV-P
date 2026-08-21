"""
AQ-CHIRPS3 v3 -- cache POR DIA, uma requisicao por dia, para sempre.

DIAGNOSTICO QUE MOTIVOU A v3 (2026-08-09):
a v2 rodou 163 minutos e gravou ZERO meses novos, com 939 falhas no `rnl` e
1.297 de 1.297 no `sat`. Antes disso a sondagem de um dia funcionava nos dois
flavors e a v1 tinha baixado 28 meses. Conclusao: o servidor da CHC passou a
recusar as requisicoes -- ele tem protecao contra scraping documentada, e nos
batemos nele com 2.594 pedidos mais retries.

O ERRO DE PROJETO QUE TORNOU ISSO IRRECUPERAVEL:
a v1 e a v2 cacheavam POR MES. Um mes so era gravado se todos os dias
chegassem. Com falhas espalhadas, quase nenhum mes fechava, entao quase nada
era cacheado, entao cada nova execucao re-requisitava TUDO -- inclusive os
dias que ja tinham dado certo. O script se auto-sabotava: quanto mais rodava,
mais requisicoes gastava, mais bloqueado ficava.

O QUE MUDA AQUI, e e a mudanca central:
cada dia vira um arquivo `.npz` minusculo (36x37 float32, ~5 KB) assim que
chega. Dia bem-sucedido e progresso PERMANENTE. Uma reexecucao pede apenas o
que falta. Mesmo que so 50 dias passem por rodada, em algumas rodadas a serie
fecha -- e nenhuma requisicao e desperdicada repetindo o que ja temos.

DEMAIS DECISOES, todas para nao reativar o bloqueio:
  - UMA conexao. Concorrencia foi o que derrubou; paralelismo aqui nao acelera,
    porque o gargalo passa a ser a tolerancia do servidor.
  - pausa de 1,5 s entre dias, e crescente apos falha.
  - `--limite N` para parar depois de N dias novos, permitindo rodadas curtas
    e espacadas em vez de uma maratona que dispara a protecao.
  - `rnl` como padrao. O `sat` e analise de sensibilidade, nao requisito;
    dobrar as requisicoes por ele foi mau negocio.
  - PARADA AUTOMATICA apos 25 falhas consecutivas: se o servidor esta
    recusando, insistir so aprofunda o bloqueio. Melhor sair e voltar depois.

Uso:
    python scripts/externo/aq_chirps3_v3.py --status
    python scripts/externo/aq_chirps3_v3.py --limite 200
    python scripts/externo/aq_chirps3_v3.py --montar     # junta o cache em NetCDF
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "90")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
RUNS = REPO / "local_runs"
STORE = RUNS / "geostore"
OUT = RUNS / "aq-chirps3"
CACHE = OUT / "cache_dia"

AOI_BNG = (350_000, 350_000, 400_000, 450_000)
BBOX = (-3.25, 52.60, -1.50, 54.40)
DIAS_ANTES = 14
PAUSA = 1.5
TENTATIVAS = 2
FALHAS_SEGUIDAS_MAX = 25


def url_dia(d: dt.date, flavor: str) -> str:
    return ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/"
            f"{flavor}/cogs/{d.year}/chirps-v3.0.{flavor}.{d.year}."
            f"{d.month:02d}.{d.day:02d}.cog")


def caminho_cache(d: dt.date, flavor: str) -> Path:
    return CACHE / flavor / f"{d.isoformat()}.npz"


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


def buscar_e_cachear(d: dt.date, flavor: str) -> bool:
    import numpy as np
    import rioxarray

    destino = caminho_cache(d, flavor)
    if destino.exists() and destino.stat().st_size > 0:
        return True
    for tentativa in range(TENTATIVAS):
        try:
            da = rioxarray.open_rasterio(url_dia(d, flavor), chunks=None,
                                         masked=True, lock=False)
            da = da.rio.clip_box(minx=BBOX[0], miny=BBOX[1],
                                 maxx=BBOX[2], maxy=BBOX[3]).load()
            arr = np.asarray(da.squeeze(), dtype="float32")
            arr = np.where(arr == -9999.0, np.nan, arr)
            destino.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destino, precip=arr,
                                y=np.asarray(da["y"].values, dtype="float64"),
                                x=np.asarray(da["x"].values, dtype="float64"))
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2.5 * (tentativa + 1))
    return False


def main() -> int:
    args = sys.argv[1:]
    flavors = ["rnl"]
    if "--flavor" in args:
        flavors = [args[args.index("--flavor") + 1]]
    if "--com-sat" in args:
        flavors = ["rnl", "sat"]
    limite = int(args[args.index("--limite") + 1]) if "--limite" in args else 10_000

    datas = datas_necessarias()

    if "--status" in args:
        print("===AQ_CHIRPS3_V3_STATUS===")
        print(f"DATAS_NECESSARIAS={len(datas)}")
        for f in ("rnl", "sat"):
            tem = sum(1 for d in datas if caminho_cache(d, f).exists())
            print(f"[{f}] em_cache={tem}/{len(datas)} ({100*tem/len(datas):.1f}%) "
                  f"faltam={len(datas)-tem}")
        print("===END===")
        return 0

    if "--montar" in args:
        import numpy as np
        import xarray as xr

        print("===AQ_CHIRPS3_V3_MONTAR===")
        for flavor in flavors:
            por_mes: dict = defaultdict(list)
            for d in datas:
                if caminho_cache(d, flavor).exists():
                    por_mes[(d.year, d.month)].append(d)
            destino_dir = OUT / flavor
            destino_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            for (ano, mes), dias in sorted(por_mes.items()):
                partes = []
                for d in dias:
                    z = np.load(caminho_cache(d, flavor))
                    partes.append(xr.Dataset(
                        {"precip": (("y", "x"), z["precip"])},
                        coords={"y": z["y"], "x": z["x"],
                                "time": np.datetime64(d)}).expand_dims("time"))
                if partes:
                    xr.concat(partes, dim="time").to_netcdf(
                        destino_dir / f"chirps3_{flavor}_{ano}-{mes:02d}.nc")
                    n += 1
            print(f"[{flavor}] meses_montados={n} de {len(por_mes)} com cache")
        print("===END===")
        return 0

    print("===AQ_CHIRPS3_V3===")
    print(f"DATAS={len(datas)} FLAVORS={flavors} limite={limite} pausa={PAUSA}s")
    for flavor in flavors:
        faltam = [d for d in datas if not caminho_cache(d, flavor).exists()]
        ja = len(datas) - len(faltam)
        print(f"[{flavor}] em_cache={ja} faltam={len(faltam)}")
        t0 = time.time()
        novos = falhas = seguidas = 0
        for d in faltam[:limite]:
            if buscar_e_cachear(d, flavor):
                novos += 1
                seguidas = 0
            else:
                falhas += 1
                seguidas += 1
                if seguidas >= FALHAS_SEGUIDAS_MAX:
                    print(f"  PARANDO: {seguidas} falhas seguidas -- servidor "
                          f"provavelmente recusando. Rode de novo mais tarde; "
                          f"os {novos} dias novos ja estao salvos.")
                    break
            time.sleep(PAUSA)
            if (novos + falhas) % 50 == 0:
                print(f"  [{flavor}] novos={novos} falhas={falhas} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
        total = sum(1 for d in datas if caminho_cache(d, flavor).exists())
        print(f"[{flavor}] novos={novos} falhas={falhas} "
              f"TOTAL_EM_CACHE={total}/{len(datas)} ({100*total/len(datas):.1f}%) "
              f"tempo={(time.time()-t0)/60:.1f}min")

    (OUT / "cache_status.json").write_text(json.dumps({
        "datas": len(datas),
        "em_cache": {f: sum(1 for d in datas if caminho_cache(d, f).exists())
                     for f in ("rnl", "sat")},
        "estrategia": "cache por dia; dia bem-sucedido e progresso permanente",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
