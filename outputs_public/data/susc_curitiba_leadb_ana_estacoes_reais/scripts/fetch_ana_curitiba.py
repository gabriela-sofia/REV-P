"""Lead B (Curitiba) -- reproduz para Curitiba o mesmo metodo real ja usado
no Lead B de Recife (`lead_b_ana_estacoes_reais.md`, SUSC-20A): consulta o
endpoint ArcGIS REST publico da ANA (`snirh.gov.br/arcgis/rest/services`)
para inventariar estacoes fluviometricas reais na regiao metropolitana de
Curitiba, depois baixa as series historicas reais de cada estacao candidata
via o endpoint legado `telemetriaws1.ana.gov.br` e verifica cobertura
temporal real (nao o metadado `ULTIMAATUALIZACAO`, que subestima, como ja
documentado no Lead B de Recife).

Isto e aquisicao real de dados publicos (rede habilitada nesta sessao,
diferente da tentativa offline-first anterior documentada em
`SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md`). Nao cria label, nao
confirma evento por si so -- so estabelece corroboracao hidrologica real,
igual ao metodo ja usado e documentado para Recife.
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
RESULTS = BUNDLE / "results"
REPORTS = BUNDLE / "reports"

# Bbox metropolitana de Curitiba (mesmo metodo espacial do Lead B Recife).
BBOX = "-49.45,-25.65,-49.10,-25.30"
INVENTORY_URL = (
    "https://www.snirh.gov.br/arcgis/rest/services/Hidroweb_BH/INVENTARIOS_ESTACOES/"
    f"MapServer/0/query?geometry={BBOX}&geometryType=esriGeometryEnvelope&inSR=4326"
    "&spatialRel=esriSpatialRelIntersects&where=EST_CD_FLU+IS+NOT+NULL"
    "&outFields=EST_CD_FLU,RIO_NM,MUN_NM,ULTIMAATUALIZACAO,OPERANDO,POSSUI_DADOS&f=json"
)
SERIE_URL_TMPL = (
    "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica"
    "?codEstacao={cod}&dataInicio=&dataFim=&tipoDados=3&nivelConsistencia="
)

# Evento-alvo conhecido (S17C_REF_0060 / CUR_2022_01_15, ja documentado no
# projeto como data oficial com fonte administrativa -- ver
# SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md).
TARGET_EVENT_DATES = ["2022-01-15", "2022-01-16"]


def fetch_inventory() -> list[dict]:
    with urlopen(INVENTORY_URL, timeout=30) as resp:
        data = json.load(resp)
    return [f["attributes"] for f in data.get("features", [])]


def fetch_series(cod: str, tipo_dados: int = 3) -> list[tuple[str, float]]:
    """Returns list of (date_iso, valor) for all real (non-empty) daily
    values across every SerieHistorica monthly row. tipo_dados=3 -> Vazao
    (flow, m3/s); tipo_dados=1 -> Cota (water level, cm). Field prefix
    differs accordingly (VazaoNN vs CotaNN)."""
    url = (
        "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica"
        f"?codEstacao={cod}&dataInicio=&dataFim=&tipoDados={tipo_dados}&nivelConsistencia="
    )
    with urlopen(url, timeout=60) as resp:
        raw = resp.read()
    root = ET.fromstring(raw)
    # <DocumentElement xmlns=""> resets the namespace for the actual data
    # rows to none, even though the outer <DataTable> is in http://MRCS/ --
    # so SerieHistorica/DataHora/VazaoNN|CotaNN are plain (no-namespace) tags.
    prefix = "Vazao" if tipo_dados == 3 else "Cota"
    out: list[tuple[str, float]] = []
    for row in root.iter("SerieHistorica"):
        data_hora = row.findtext("DataHora")
        if not data_hora:
            continue
        year, month = int(data_hora[:4]), int(data_hora[5:7])
        for day in range(1, 32):
            tag = f"{prefix}{day:02d}"
            val = row.findtext(tag)
            if val in (None, ""):
                continue
            try:
                v = float(val)
            except ValueError:
                continue
            try:
                iso = f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                continue
            out.append((iso, v))
    out.sort()
    return out


def run() -> None:
    stations = fetch_inventory()
    print(f"[lead_b_curitiba] n_stations_bbox={len(stations)}")

    # Mesmo criterio amplo ja usado no Lead B de Recife (nao restringir ao
    # municipio-sede -- estacoes upstream/vizinhas tambem corroboram).
    candidates = [s for s in stations if s.get("OPERANDO") == "Sim" and s.get("POSSUI_DADOS") == "Sim"]
    print(f"[lead_b_curitiba] n_stations_bbox_operando_com_dados={len(candidates)}")

    rows_out = []
    for s in candidates:
        cod = s["EST_CD_FLU"]
        best_series: list[tuple[str, float]] = []
        best_tipo = ""
        for tipo, label in ((3, "vazao"), (1, "cota")):
            try:
                series = fetch_series(cod, tipo_dados=tipo)
            except Exception:
                series = []
            time.sleep(0.2)
            if len(series) > len(best_series):
                best_series, best_tipo = series, label
        n = len(best_series)
        d_ini = best_series[0][0] if best_series else ""
        d_fim = best_series[-1][0] if best_series else ""
        dates_set = {d for d, _ in best_series}
        covers_event = any(d in dates_set for d in TARGET_EVENT_DATES)
        row = {"cod_estacao": cod, "rio": s.get("RIO_NM", ""), "municipio": s.get("MUN_NM", ""),
               "tipo_usado": best_tipo, "n_dias_com_dado": n, "data_inicio": d_ini, "data_fim": d_fim,
               "cobre_evento_2022_01_15": covers_event, "erro": ""}
        if covers_event:
            vals = np.array([v for _, v in best_series])
            p95 = float(np.percentile(vals, 95))
            event_vals = {d: v for d, v in best_series if d in dates_set and d in TARGET_EVENT_DATES}
            row["p95_serie_completa"] = round(p95, 2)
            row["valor_evento"] = json.dumps(event_vals)
            row["evento_acima_p95"] = any(v >= p95 for v in event_vals.values())
        rows_out.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    fieldnames = ["cod_estacao", "rio", "municipio", "tipo_usado", "n_dias_com_dado",
                  "data_inicio", "data_fim", "cobre_evento_2022_01_15",
                  "p95_serie_completa", "valor_evento", "evento_acima_p95", "erro"]
    import csv
    with (RESULTS / "ana_curitiba_series_coverage.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows_out, key=lambda r: r["data_fim"], reverse=True):
            w.writerow({k: r.get(k, "") for k in fieldnames})

    with (RESULTS / "ana_curitiba_full_inventory.json").open("w", encoding="utf-8") as fh:
        json.dump(stations, fh, indent=2, ensure_ascii=False)

    covering = [r for r in rows_out if r.get("cobre_evento_2022_01_15")]
    most_recent = sorted(rows_out, key=lambda r: r["data_fim"], reverse=True)[:5]
    print(f"[lead_b_curitiba] n_estacoes_cobrindo_evento_2022_01_15={len(covering)}")
    for r in covering:
        print(f"  COBRE EVENTO: {r['cod_estacao']} ({r['rio']}): p95={r.get('p95_serie_completa')} "
              f"valor_evento={r.get('valor_evento')} acima_p95={r.get('evento_acima_p95')}")
    print("[lead_b_curitiba] 5 estacoes com dado mais recente (mesmo sem cobrir o evento):")
    for r in most_recent:
        print(f"  {r['cod_estacao']} ({r['rio']}, {r['municipio']}): {r['data_inicio']} a {r['data_fim']} "
              f"(tipo={r['tipo_usado']}, n={r['n_dias_com_dado']})")


if __name__ == "__main__":
    run()
