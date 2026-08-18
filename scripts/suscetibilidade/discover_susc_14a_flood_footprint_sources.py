"""
SUSC-14A FASE 6a - discover public flood-footprint (vector) sources.

Builds a deterministic registry of well-known public flood-footprint providers
(Copernicus EMS Rapid Mapping, Dartmouth Flood Observatory, Global Flood
Database, SGB/CPRM, INEA, Defesa Civil RJ, APAC, GeoCuritiba) with the event /
region / activation they relate to. No raster is targeted: only vector packages
(GeoJSON/SHP/KML/GPKG/WKT or PDF maps with vector/table). With SUSC_13B_NETWORK=1
each catalog root may be probed; offline everything records network_disabled.

Writes:
  - manifests/suscetibilidade/susc_14a_flood_footprint_source_registry_v1.csv
  - outputs_public/suscetibilidade/SUSC_14A_flood_footprint_discovery_report.md
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import rel, write_csv, write_markdown  # noqa: E402

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_14a_flood_footprint_source_registry_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_flood_footprint_discovery_report.md"

FIELDS = [
    "footprint_source_id", "region", "provider", "product", "event_reference",
    "event_period", "source_url", "expected_format", "is_vector_expected",
    "raster_only_risk", "access_method", "priority", "download_candidate",
    "review_only", "notes",
]

# provider, region, product, event_reference, event_period, url, format, vector_expected, raster_only_risk
SOURCES = [
    ("petropolis", "Copernicus EMS Rapid Mapping", "EMSR Activation (Petropolis flood 2022)",
     "Petropolis enxurrada/deslizamento 2022", "2022-02-15/2022-02-18",
     "https://emergency.copernicus.eu/mapping/list-of-activations-rapid", "geojson|shp|kml", 1, 0),
    ("petropolis", "Dartmouth Flood Observatory", "DFO Brazil flood catalog",
     "Petropolis/Rio de Janeiro 2022", "2022",
     "https://floodobservatory.colorado.edu/", "shp|kml", 1, 1),
    ("petropolis", "INEA-RJ", "Areas inundaveis / setores afetados 2022",
     "Petropolis 2022", "2022", "https://inea.maps.arcgis.com", "geojson|arcgis", 1, 0),
    ("petropolis", "Defesa Civil RJ", "Area atingida Petropolis 2022 (relatorio/mapa)",
     "Petropolis 2022", "2022", "https://www.defesacivil.rj.gov.br", "pdf|geojson", 1, 1),
    ("petropolis", "SGB/CPRM", "Inundacao / setores de risco shapefile",
     "Petropolis serrana RJ", "", "https://geoportal.sgb.gov.br/server/rest/services", "shp|geojson|arcgis", 1, 0),
    ("recife", "APAC Pernambuco", "Areas alagadas / manchas de inundacao Recife",
     "Recife eventos urbanos", "", "https://www.apac.pe.gov.br", "geojson|shp|pdf", 1, 1),
    ("recife", "Global Flood Database", "GFD historical flood extent (vector export)",
     "Recife/PE historical", "", "http://global-flood-database.cloudtostreet.ai/", "geojson", 1, 1),
    ("recife", "NASA / MODIS NRT Global Flood", "Flood event vector (where published)",
     "Brazil NE", "", "https://www.earthobservatory.nasa.gov/", "geojson", 0, 1),
    ("curitiba", "GeoCuritiba / IPPUC", "Pontos de alagamento / areas alagaveis",
     "Curitiba urbano", "", "https://geocuritiba.ippuc.org.br/server/rest/services", "geojson|arcgis", 1, 0),
    ("curitiba", "IAT / Aguas Parana", "Manchas de inundacao / bacias",
     "Bacia do Iguacu / Curitiba", "", "https://geoportal.iat.pr.gov.br/geoserver/ows", "geojson|wfs", 1, 0),
]


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Flood Footprint Source Discovery")
    print("=" * 60)
    net = g14.network_enabled()
    rows: list[dict] = []
    for i, (region, provider, product, evref, period, url, fmt, vec, raster_risk) in enumerate(SOURCES):
        method = "arcgis_rest" if "arcgis" in fmt or "rest/services" in url else (
            "wfs_getcapabilities" if "wfs" in fmt or "ows" in url else "data_portal_html")
        probe = "probe=network_disabled" if not net else "probe=html_root_not_fetched"
        if net and method in ("arcgis_rest", "wfs_getcapabilities"):
            res = g14.http_get(url, accept="application/json")
            probe = f"probe={res.get('status','')}"
        rows.append({
            "footprint_source_id": f"S14AFP_{i:03d}", "region": region, "provider": provider,
            "product": product, "event_reference": evref, "event_period": period,
            "source_url": url, "expected_format": fmt,
            "is_vector_expected": "true" if vec else "false",
            "raster_only_risk": "true" if raster_risk else "false",
            "access_method": method,
            "priority": "high" if (vec and period) else ("medium" if vec else "low"),
            "download_candidate": "true" if vec else "false",
            "review_only": "true",
            "notes": f"fonte publica de footprint vetorial; sem raster pesado; {probe}",
        })
    write_csv(REGISTRY, rows, FIELDS)

    by_region = Counter(r["region"] for r in rows)
    vec = sum(1 for r in rows if r["is_vector_expected"] == "true")
    raster_risk = sum(1 for r in rows if r["raster_only_risk"] == "true")

    md = f"""# SUSC-14A - Descoberta de footprints publicos de cheia

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

## 1. Objetivo
Localizar footprints publicos de area alagada/inundada em formato vetorial
(GeoJSON/SHP/KML/GPKG/WKT ou PDF com vetor/tabela) para Recife, Petropolis e
Curitiba, priorizando a ativacao Copernicus EMS de Petropolis 2022. **Nenhum
raster pesado e baixado**: se so houver raster/preview, registra-se
`footprint_unavailable_vector_required`.

## 2. Politica de rede
- Ativacao live: `SUSC_13B_NETWORK=1`; rede nesta execucao: **{"Sim" if net else "Nao"}**.
- Sem Google Maps, sem geocoding, sem chave de API, sem raster Sentinel.

## 3. Fontes registradas
Total: **{len(rows)}** | com expectativa de vetor: **{vec}** | com risco de so-raster: **{raster_risk}**.

| regiao | fontes |
|---|---|
{chr(10).join(f"| {k} | {v} |" for k, v in sorted(by_region.items())) or "| nenhum | 0 |"}

## 4. Petropolis 2022
Copernicus EMS Rapid Mapping e a fonte de maior prioridade (ativacao com periodo
explicito 2022-02). INEA-RJ, Defesa Civil RJ e SGB/CPRM complementam com setores
afetados/areas inundaveis.

## 5. Governanca
Fontes sao raizes/catalogos institucionais; nenhum URL de download direto e
inventado. Tudo review-only; nada vira ground truth, treino ou score v7.
"""
    write_markdown(REPORT, md)
    print(f"registry -> {rel(REGISTRY)} ({len(rows)} sources; {vec} vector-expected)")
    print("review-only. No raster. No invented direct URL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
