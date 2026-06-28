"""SUSC-14B-LIVE FASE 1 - discover official spatial references.

Live, fail-closed search over official CKAN/ArcGIS/WFS/HTML endpoints for street
segments, addresses, neighborhoods, drainage and flood points. No generic
geocoding, no Google Maps, no invented coordinates.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import write_csv, write_markdown  # noqa: E402

os.environ.setdefault(wd.NETWORK_ENV, "1")
wd.TIMEOUT = 8
wd.RETRIES = 1

REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_14b_official_spatial_reference_registry_v1.csv"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14B_official_spatial_reference_discovery_report.md"

FIELDS = [
    "reference_id", "region", "institution", "source_url", "source_title", "source_type",
    "format", "access_method", "status_code", "contains_street_segments",
    "contains_address_points", "contains_neighborhood_polygons", "contains_drainage",
    "contains_flood_points", "contains_flood_polygons", "download_candidate",
    "priority", "reason", "allowed_for_training", "can_be_ground_truth",
    "can_be_used_as_ground_truth", "review_only",
]

RECIFE_QUERIES = [
    "logradouros Recife", "eixos logradouros Recife", "eixo viario Recife",
    "enderecos Recife", "cadastro logradouro Recife", "bairros Recife geojson",
    "limites bairros Recife", "pontos criticos alagamento Recife", "drenagem Recife",
    "canais Recife", "rios Recife", "hidrografia Recife", "malha viaria Recife",
    "trechos de via Recife",
]
CURITIBA_QUERIES = ["arruamento", "eixo de rua", "logradouros", "bairros", "hidrografia", "drenagem", "pontos de alagamento"]
PETROPOLIS_QUERIES = ["bairros", "logradouros", "setores atingidos", "areas atingidas 2022", "hidrografia"]

CKAN_ENDPOINTS = [
    ("recife", "Prefeitura do Recife - Dados Abertos", "official", "http://dados.recife.pe.gov.br/api/3/action/package_search", RECIFE_QUERIES),
    ("curitiba", "Curitiba - Dados Abertos", "official", "https://dadosabertos.curitiba.pr.gov.br/api/3/action/package_search", CURITIBA_QUERIES),
    ("petropolis", "Governo do RJ - Dados Abertos", "state", "http://www.dados.rj.gov.br/api/3/action/package_search", PETROPOLIS_QUERIES),
]
SERVICE_ENDPOINTS = [
    ("recife", "ESIG Recife", "official", "https://www.recife.pe.gov.br/arcgis/rest/services", "arcgis_rest"),
    ("curitiba", "GeoCuritiba/IPPUC", "official", "https://geocuritiba.ippuc.org.br/server/rest/services", "arcgis_rest"),
    ("curitiba", "IAT/Aguas Parana", "state", "https://geoportal.iat.pr.gov.br/geoserver/ows", "wfs_getcapabilities"),
    ("petropolis", "CPRM/SGB GeoSGB", "federal", "https://geoportal.sgb.gov.br/server/rest/services", "arcgis_rest"),
    ("petropolis", "Prefeitura de Petropolis", "official", "https://www.petropolis.rj.gov.br", "data_portal_html"),
]

STREET_TERMS = ("logradouro", "logradouros", "eixo", "eixos", "rua", "vias", "malha viaria", "arruamento", "trecho")
ADDRESS_TERMS = ("endereco", "enderecos", "numero", "cadastro")
HOOD_TERMS = ("bairro", "bairros", "limite", "limites")
DRAINAGE_TERMS = ("drenagem", "canal", "canais", "hidrografia", "rio", "rios")
FLOOD_TERMS = ("alagamento", "inundacao", "enchente", "ponto critico", "area atingida", "areas atingidas")
VECTOR_EXTS = {".csv", ".xlsx", ".xls", ".geojson", ".json", ".kml", ".kmz", ".zip", ".gpkg", ".gml", ".xml"}


def _contains(text: str, terms: tuple[str, ...]) -> str:
    low = g14.normalize_text(text)
    return str(any(g14.normalize_text(t) in low for t in terms)).lower()


def _fmt(url: str, fmt_hint: str = "") -> str:
    low = (url or "").lower().split("?")[0]
    for ext in sorted(VECTOR_EXTS, key=len, reverse=True):
        if low.endswith(ext):
            return ext.lstrip(".")
    return (fmt_hint or "api").lower()


def _download_candidate(url: str, title: str, fmt_hint: str = "") -> str:
    fmt = _fmt(url, fmt_hint)
    text = f"{title} {url}".lower()
    if any(block in text for block in (".tif", ".tiff", ".jp2", ".img", ".pdf", "sentinel")):
        return "false"
    return str(fmt in {e.lstrip(".") for e in VECTOR_EXTS} or "query" in url or "FeatureServer" in url or "MapServer" in url).lower()


def _priority(title: str, url: str) -> str:
    text = f"{title} {url}"
    if _contains(text, STREET_TERMS) == "true" or _contains(text, ADDRESS_TERMS) == "true":
        return "1"
    if _contains(text, HOOD_TERMS) == "true":
        return "2"
    if _contains(text, DRAINAGE_TERMS + FLOOD_TERMS) == "true":
        return "3"
    return "4"


def _row(region, inst, stype, url, title, access, status, fmt="", reason=""):
    text = f"{title} {url}"
    return {
        "reference_id": "",
        "region": region,
        "institution": inst,
        "source_url": url,
        "source_title": title[:180],
        "source_type": stype,
        "format": _fmt(url, fmt),
        "access_method": access,
        "status_code": status,
        "contains_street_segments": _contains(text, STREET_TERMS),
        "contains_address_points": _contains(text, ADDRESS_TERMS),
        "contains_neighborhood_polygons": _contains(text, HOOD_TERMS),
        "contains_drainage": _contains(text, DRAINAGE_TERMS),
        "contains_flood_points": _contains(text, FLOOD_TERMS),
        "contains_flood_polygons": _contains(text, ("mancha", "poligono", "area atingida", "areas atingidas")),
        "download_candidate": _download_candidate(url, title, fmt),
        "priority": _priority(title, url),
        "reason": reason or "official_live_discovery_review_only",
        "allowed_for_training": "false",
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "review_only": "true",
    }


def _ckan_rows(region, inst, stype, endpoint, queries):
    rows = []
    for query in queries:
        url = endpoint + "?" + urlencode({"q": query, "rows": 20})
        rec = wd.http_get(url)
        status = str(rec.get("http_code") or rec.get("status") or "")
        body = rec.get("bytes") or b""
        if not body:
            rows.append(_row(region, inst, stype, endpoint, query, "ckan_api", status, "api", "ckan_probe_no_body_or_offline"))
            continue
        try:
            data = json.loads(body.decode("utf-8", "ignore"))
        except ValueError:
            rows.append(_row(region, inst, stype, endpoint, query, "ckan_api", status, "api", "ckan_json_parse_failed"))
            continue
        for pkg in (data.get("result") or {}).get("results", [])[:20]:
            ptitle = pkg.get("title") or pkg.get("name") or query
            for res in pkg.get("resources", []) or []:
                rurl = res.get("url") or ""
                if not rurl:
                    continue
                title = f"{ptitle} - {res.get('name') or res.get('description') or res.get('format') or 'resource'}"
                if not any(_contains(title + " " + rurl, terms) == "true" for terms in (STREET_TERMS, ADDRESS_TERMS, HOOD_TERMS, DRAINAGE_TERMS, FLOOD_TERMS)):
                    continue
                rows.append(_row(region, inst, stype, rurl, title, "ckan_resource", status, res.get("format", ""), f"query={query}"))
    return rows


def _service_rows():
    rows = []
    for region, inst, stype, url, access in SERVICE_ENDPOINTS:
        probe = url
        if access == "arcgis_rest":
            probe = url.rstrip("/") + "?f=pjson"
        elif access == "wfs_getcapabilities":
            probe = url + "?service=WFS&request=GetCapabilities"
        rec = wd.http_get(probe)
        status = str(rec.get("http_code") or rec.get("status") or "")
        rows.append(_row(region, inst, stype, probe, inst, access, status, "api", "official_service_root_probe"))
    return rows


def main() -> int:
    print("=" * 60)
    print("SUSC-14B-LIVE Official Spatial Reference Discovery")
    print("=" * 60)
    rows = []
    for endpoint in CKAN_ENDPOINTS:
        rows.extend(_ckan_rows(*endpoint))
    rows.extend(_service_rows())
    # Baseline inherited from SUSC-14A: already parsed official local references.
    rows.append(_row("recife", "SUSC-14A local official baseline", "official",
                     "datasets/suscetibilidade/susc_14a_official_geocoding_reference_features_v1.csv",
                     "SUSC-14A official geocoding reference features", "local_reuse", "local", "csv",
                     "baseline oficial rastreavel herdado do 14A"))
    dedup = {}
    for row in rows:
        key = (row["region"], row["source_url"], row["source_title"])
        dedup[key] = row
    rows = sorted(dedup.values(), key=lambda r: (r["region"], r["priority"], r["source_title"]))
    for idx, row in enumerate(rows):
        row["reference_id"] = f"S14BREF_{idx:04d}"
    write_csv(REGISTRY, rows, FIELDS)
    by_region = {}
    for row in rows:
        by_region[row["region"]] = by_region.get(row["region"], 0) + 1
    candidates = sum(1 for row in rows if row["download_candidate"] == "true")
    write_markdown(REPORT, f"""# SUSC-14B - descoberta live de referencias espaciais oficiais

Status: review-only. `allowed_for_training=false`; `can_be_ground_truth=false`.

- Network gate: `{wd.NETWORK_ENV}` = `{wd.network_enabled()}`
- Referencias registradas: **{len(rows)}**
- Por regiao: **{by_region}**
- Download candidates: **{candidates}**

A busca priorizou Recife, mas tambem registrou Curitiba e Petropolis. Apenas
fontes oficiais/rastreaveis foram registradas. Nenhum Google Maps, Nominatim ou
geocoding generico foi usado. O baseline local do SUSC-14A foi mantido como
referencia rastreavel para comparacao e reproducibilidade.
""")
    print(f"registry -> {REGISTRY.relative_to(ROOT)} ({len(rows)} refs; candidates {candidates})")
    print("review-only. No ground truth. No training. No generic geocoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
