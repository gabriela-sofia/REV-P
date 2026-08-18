"""SUSC-15A FASE 3 - discover official precision reference sources."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, write_csv, write_markdown  # noqa: E402

MAN = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
REGISTRY_14B = MAN / "susc_14b_official_spatial_reference_registry_v1.csv"
REGISTRY = MAN / "susc_15a_precision_reference_source_registry_v1.csv"
REPORT = OUT / "SUSC_15A_precision_reference_discovery_report.md"

FIELDS = [
    "source_id", "region", "institution", "url", "source_type", "format", "access_method",
    "contains_event_point", "contains_event_polygon", "contains_address_point", "contains_parcel",
    "contains_house_number_range", "contains_intersection", "contains_street_segment",
    "contains_flood_footprint", "priority", "download_candidate", "reason", "review_only",
]

SEEDS = [
    ("recife", "Prefeitura do Recife - ESIG", "https://www.recife.pe.gov.br/arcgis/rest/services", "official", "arcgis_rest", "json", 1,
     "ESIG Recife FeatureServer logradouro/endereco/alagamento"),
    ("recife", "Prefeitura do Recife - Dados Abertos", "http://dados.recife.pe.gov.br/api/3/action/package_search", "official", "ckan_api", "json", 2,
     "enderecamento, pontos criticos, logradouros e bairros"),
    ("curitiba", "GeoCuritiba / IPPUC", "https://geocuritiba.ippuc.org.br/server/rest/services", "official", "arcgis_rest", "json", 1,
     "GeoCuritiba eixos, enderecos, pontos de alagamento"),
    ("curitiba", "Curitiba Dados Abertos", "https://dadosabertos.curitiba.pr.gov.br/api/3/action/package_search", "official", "ckan_api", "json", 2,
     "SIAC 156, alagamento e logradouros"),
    ("petropolis", "CPRM/SGB GeoSGB", "https://geoportal.sgb.gov.br/server/rest/services", "federal", "arcgis_rest", "json", 1,
     "setores, hidrografia e areas atingidas rastreaveis"),
    ("petropolis", "INEA-RJ", "https://inea.maps.arcgis.com", "state", "arcgis_rest", "html", 3,
     "vetores estaduais de inundacao/areas atingidas"),
    ("petropolis", "Copernicus EMS", "https://emergency.copernicus.eu/mapping/list-of-components/EMSR?", "technical", "portal_html", "html", 4,
     "footprints tecnicos rastreaveis para ativacoes de inundacao"),
    ("petropolis", "Dartmouth Flood Observatory", "https://floodobservatory.colorado.edu/Archives/index.html", "technical", "portal_html", "html", 5,
     "inventario tecnico de footprints de inundacao"),
]


def _flags(text: str) -> dict:
    low = g14.normalize_text(text)
    return {
        "contains_event_point": any(k in low for k in ("ocorrencia", "ponto", "alagamento", "defesa civil")),
        "contains_event_polygon": any(k in low for k in ("poligono", "area atingida", "inundacao", "footprint")),
        "contains_address_point": any(k in low for k in ("endereco", "enderecamento", "cadastro")),
        "contains_parcel": any(k in low for k in ("lote", "parcel", "cadastro imobiliario")),
        "contains_house_number_range": any(k in low for k in ("numeracao", "faixa", "house number")),
        "contains_intersection": any(k in low for k in ("intersec", "cruzamento", "esquina")),
        "contains_street_segment": any(k in low for k in ("logradouro", "eixo", "rua", "via")),
        "contains_flood_footprint": any(k in low for k in ("alagamento", "inundacao", "flood", "footprint", "area atingida")),
    }


def _candidate(fmt: str, reason: str) -> str:
    low = reason.lower()
    if fmt in {"geojson", "json", "csv", "xlsx", "kml", "zip", "gpkg", "wkt", "gml"}:
        return "true"
    if any(k in low for k in ("html", "portal")):
        return "false"
    return "true"


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Precision Reference Discovery")
    print("=" * 60)
    rows = []
    for idx, seed in enumerate(SEEDS):
        region, inst, url, source_type, access_method, fmt, priority, reason = seed
        flags = _flags(f"{inst} {reason}")
        row = {
            "source_id": f"S15ASRC_{idx:04d}", "region": region, "institution": inst, "url": url,
            "source_type": source_type, "format": fmt, "access_method": access_method,
            "priority": priority, "download_candidate": _candidate(fmt, reason),
            "reason": reason, "review_only": "true",
        }
        row.update({k: str(v).lower() for k, v in flags.items()})
        rows.append(row)
    offset = len(rows)
    for idx, src in enumerate(read_csv(REGISTRY_14B) if REGISTRY_14B.exists() else []):
        text = f"{src.get('source_title','')} {src.get('source_url','')} {src.get('institution','')}"
        flags = _flags(text)
        if not any(flags.values()):
            continue
        fmt = (src.get("format") or "json").lower()
        row = {
            "source_id": f"S15ASRC_{offset + idx:04d}",
            "region": src.get("region", ""),
            "institution": src.get("institution", ""),
            "url": src.get("source_url", ""),
            "source_type": src.get("source_type", "official"),
            "format": fmt,
            "access_method": src.get("access_method", ""),
            "priority": src.get("priority", "5"),
            "download_candidate": _candidate(fmt, text),
            "reason": f"herdado do registro oficial 14B: {src.get('source_title','')}",
            "review_only": "true",
        }
        row.update({k: str(v).lower() for k, v in flags.items()})
        rows.append(row)
    write_csv(REGISTRY, rows, FIELDS)
    write_markdown(REPORT, f"""# SUSC-15A - descoberta de fontes oficiais de precisao

Status: review-only. Fontes registradas para busca de ponto/poligono de evento,
enderecamento oficial, lote, intersecao, linear referencing e footprint.

- Fontes registradas: **{len(rows)}**
- Candidatas a download direto: **{sum(1 for r in rows if r['download_candidate'] == 'true')}**

Nenhuma fonte generica de geocoding foi registrada como evidencia. OSM/Nominatim
nao e usado como fonte de evidencia.
""")
    print(f"sources: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
